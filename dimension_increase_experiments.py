import os
import torch
import numpy as np
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt
import torchquad

from utils.metrics import compute_log_normalizing_constant
from utils.score_estimators import get_score_function
from sde_lib import get_sde

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_gmm_dimension(D, num_modes,device):
    setup_seed(D)
    c = torch.ones(num_modes, device=device)/D
    means = torch.randn((num_modes,D), device=device) * 2
    means+= torch.ones_like(means) * 3
    variances = torch.eye(D,device=device).unsqueeze(0).expand((num_modes,D,D))
    variances = variances * (torch.rand((num_modes,1,1),device=device) + 0.3 ) # random variances in range [.3, 1.3]
    gaussians = [utils.densities.MultivariateGaussian(means[i],variances[i]) for i in range(c.shape[0])]
    return utils.densities.MixtureDistribution(c,gaussians)

def get_double_well(D):
    return utils.densities.DoubleWell(D,4.)
 
def get_double_well_log_normalizing_constant(D):
    dw = utils.densities.DoubleWell(1,4.)
    def unnormalized_dens(x):
        return torch.exp(dw.log_prob(x))
    return D * torchquad.Boole().integrate(unnormalized_dens, 1, N=2001, integration_domain=[[-15.,15.]]).log()

def get_diff_log_z(config, dist, true, device):
    with torch.autograd.detect_anomaly(check_nan=True):
        sde = get_sde(config)
        model = get_score_function(config, dist,sde,device)
        log_z = compute_log_normalizing_constant(dist,sde,model,False)
        print(true, log_z)
        return (true-log_z).abs().cpu().detach().numpy()

def compute_statistic(distribution : utils.densities.MixtureDistribution, samples):
    f = 0
    for dist in distribution.distributions:
        f += torch.mean(torch.sum((samples-dist.mean)**2,dim=-1),dim=0)
    return f

def get_method_names(config):
    num_methods = 1 + len(config.methods_to_run) + len(config.baselines)
    method_names = [''] * num_methods
    method_names[0] = 'Ground Truth'
    k = 1
    for method in config.methods_to_run:
        method_names[k] = method
        k+=1
    for method in config.baselines:
        method_names[k] = method    
        k+=1
         
    return num_methods, method_names     


def eval(config):
    setup_seed(1)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods, method_names = get_method_names(config)
    dimensions = np.arange(1,10,step=1)
    print(dimensions)
    num_dims = len(dimensions)
    stats = np.zeros([num_methods, num_dims],dtype='double')
    w2_stats = np.zeros([num_methods, num_dims],dtype='double')
    
    num_modes = 5
    
    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)

    if not config.load_from_ckpt:
        for i, d in enumerate(dimensions):
            config.dimension = d
            distribution = get_gmm_dimension(d,num_modes,device)
            # distribution = get_double_well(d)
            # Baseline
            true_samples = distribution.sample(tot_samples)
            stats[0][i] = compute_statistic(distribution, true_samples)
            k = 1
            for method in config.methods_to_run:
                print(f'{method} {d}')
                if method == 'ZOD-MC':
                    # Rejection
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.T = 2
                    config.num_estimator_batches = 10 * d 
                    config.num_estimator_samples = 10000
                    config.sampling_eps = 5e-3
                elif method == 'RDMC':
                    # Reverse Diffusion Monte Carlo
                    distribution.keep_minimizer = False
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula'
                    config.T = 2
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = 1000
                    config.num_sampler_iterations = 100
                    config.ula_step_size = 0.1     
                    config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.T = 2
                    config.num_estimator_batches = 1
                    config.num_recursive_steps = 3
                    config.num_estimator_samples = 10
                    config.num_sampler_iterations = 5
                    config.ula_step_size = 0.1
                    config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
                    
                generated_samples = sample.sample(config,distribution)
                # stats[k][i] = get_diff_log_z(config,distribution, get_double_well_log_normalizing_constant(d),device)
                stats[k][i] = compute_statistic(distribution, generated_samples)
                w2_stats[k][i] = utils.metrics.get_w2(generated_samples,true_samples).detach().item()
                print('Stats ', stats[k,i],w2_stats[k,i])
                k+=1
    else:
        stats = torch.load(os.path.join(folder,'log_z.pt'))#.cpu().numpy()
        w2_stats = torch.load(os.path.join(folder,'w2.pt'))#.cpu().numpy()
        method_names = np.load(os.path.join(folder,'method_names.npy'))
    
    # Save method names and samples
    torch.save(stats,os.path.join(folder,'log_z.pt'))
    torch.save(w2_stats,os.path.join(folder,'w2.pt'))
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    plt.rcParams.update({
        'font.size': 14,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']
    print(stats)
    for i,method in enumerate(method_names):
        method_label = method[0].upper() + method[1:]
        if method[-2:] != 'MC':
            continue
        print(method)
        ax1.plot(dimensions,np.abs(stats[i]-stats[0]),label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax2.plot(dimensions,w2_stats[i],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    ax1.set_xlabel('Dimension')
    
    # ax1.set_ylabel(r'$|\Delta \log Z|$')
    ax1.set_ylabel(r'Error in estimation of $\mathbb{E}[f(x)]$')
    ax1.set_ylim(0,800)
    ax1.legend(loc='upper left')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('W2')
    ax2.legend(loc='upper left')
    fig.savefig(os.path.join(folder,'dimension_mmd_results.pdf'),bbox_inches='tight')


        