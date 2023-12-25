import torch
import numpy as np
import wandb
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.mmd
import sample
import matplotlib.pyplot as plt
import samplers.ula
import samplers.proximal_sampler
from math import pi, sin , cos



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_gmm_radius(K,R,device):
    sigma = 1
    c = torch.ones(K,device=device)/K
    circle = torch.tensor([[cos(2*pi*i/K),sin(2*pi*i/K)] for i in range(K)]) \
        .to(dtype=torch.double, device=device) 
    offset = torch.tensor([2.,2.],dtype=torch.double, device=device)
    means = R * (circle + offset)
    variances = torch.cat([torch.eye(2).unsqueeze(0) * sigma for i in range(K)],dim=0) \
        .to(dtype=torch.double, device=device)
    gaussians = [utils.densities.MultivariateGaussian(means[i],variances[i]) for i in range(K)]
    return utils.densities.MixtureDistribution(c,gaussians)

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
    mmd = utils.mmd.MMDLoss()

    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods, method_names = get_method_names(config)
    radiuses = np.arange(2,20,step=2)
    num_rad = len(radiuses)
    mmd_stats = np.zeros([num_methods, num_rad],dtype='double')
    samples_all = torch.zeros([num_methods, num_rad,tot_samples, config.dimension],device=device,dtype=torch.double)
    samples_all = torch.load(f'radius.pt').to(device=device).to(dtype=torch.double)
    
    print(method_names)
    for i, r in enumerate(radiuses):
        distribution = get_gmm_radius(6,r,device)
        # Baseline
        samples_all[0][i] = distribution.sample(tot_samples)
        k = 1
        for method in config.methods_to_run:
            if method == 'rejection':
                # Rejection
                distribution.keep_minimizer = True
                config.score_method = 'p0t'
                config.p0t_method = 'rejection'
                config.T = 10
                config.num_estimator_batches = 5
                config.num_estimator_samples = 1000
                config.sampling_eps = 5e-3
            elif method == 'ula': 
                # Reverse Diffusion Monte Carlo
                distribution.keep_minimizer = False
                config.score_method = 'p0t'
                config.p0t_method = 'ula'
                config.T = 2 if r < 6 else 3
                config.num_estimator_samples = 1000
                config.num_sampler_iterations = 100
                config.ula_step_size = 0.1     
                config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
            elif method == 'quotient-estimator':
                # Quotient-Estimator
                distribution.keep_minimizer = False
                config.score_method = 'quotient-estimator'
                config.T = 3
                config.num_estimator_batches = 5
                config.num_estimator_samples = 1000
                config.sampling_eps = 5e-3
            samples_all[k][i] = sample.sample(config,distribution)
            mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
            k+=1
            
        for method in config.baselines:
            in_cond = torch.randn_like(samples_all[0][i])
            if method == 'langevin':
                # Langevin
                distribution.keep_minimizer = False
                ula_step_size = 0.1
                num_steps_lang = 5000 
                samples_all[k][i] = samplers.ula.get_ula_samples(in_cond,
                                                                distribution.grad_log_prob,
                                                                ula_step_size,num_steps_lang,display_pbar=False)
            elif method == 'proximal': 
                # Proximal
                samples_all[k][i] = samplers.proximal_sampler.get_samples(in_cond,
                                                                        distribution,
                                                                        config.proximal_M,
                                                                        config.proximal_num_iters,
                                                                        1,device
                                                                        ).squeeze(1)
            mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
            k+=1
        xlim = [2*r - 3 * r, 2*r + 3 * r]
        ylim = [2*r - 3 * r, 2*r + 3 * r]
        fig = utils.plots.plot_all_samples(samples_all[:,i,:,:],
                                        method_names,
                                        xlim,ylim,distribution.log_prob)
        fig.savefig(f'plots/Radius_{r}.png', bbox_inches='tight')
        plt.close(fig)

    # Save MMD Information
    torch.save(samples_all, f'radius.pt')
    fig, ax = plt.subplots()
    for i,method in enumerate(method_names):
        if method == 'Ground Truth':
            continue
        ax.plot(radiuses,mmd_stats[i],label=method)
    ax.set_title('MMD as a function of mode separation')
    ax.set_xlabel('Radius')
    ax.set_ylabel('MMD')
    ax.legend()
    fig.savefig('plots/radius_mmd_results.png')


        