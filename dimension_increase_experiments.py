import os
import torch
import numpy as np
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_gmm_dimension(D, num_modes,device):
    setup_seed(D)
    c = torch.ones(num_modes, device=device)/D
    means = torch.randn((num_modes,D), device=device) * 3
    variances = torch.eye(D,device=device).unsqueeze(0).expand((num_modes,D,D))
    variances = variances * (torch.rand((num_modes,1,1),device=device) + 0.3 ) # random variances in range [.3, 1.3]
    gaussians = [utils.densities.MultivariateGaussian(means[i],variances[i]) for i in range(c.shape[0])]
    # return utils.densities.DoubleWell(D,1.)
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
    mmd = utils.metrics.MMDLoss()

    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods, method_names = get_method_names(config)
    dimensions = np.arange(1,10,step=1)
    print(dimensions)
    num_dims = len(dimensions)
    mmd_stats = np.zeros([num_methods, num_dims],dtype='double')
    w2_stats = np.zeros([num_methods, num_dims],dtype='double')
    
    num_modes = 5
    
    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)

    if not config.load_from_ckpt:
        for i, d in enumerate(dimensions):
            config.dimension = d
            distribution = get_gmm_dimension(d,num_modes,device)
            # Baseline
            true_samples = distribution.sample(tot_samples)
            k = 1
            for method in config.methods_to_run:
                print(f'{method} {d}')
                if method == 'ZOD-MC':
                    # Rejection
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.T = 5
                    config.num_estimator_batches = 10 * d 
                    config.num_estimator_samples = 10000
                    config.sampling_eps = 5e-3
                elif method == 'RDMC':
                    # Reverse Diffusion Monte Carlo
                    distribution.keep_minimizer = False
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula'
                    config.T = 2
                    config.num_estimator_samples = 1000
                    config.num_sampler_iterations = 100
                    config.ula_step_size = 0.01     
                    config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.T = 2
                    config.num_recursive_steps = 3
                    config.num_estimator_samples = 10
                    config.num_sampler_iterations = 5
                    config.ula_step_size = 0.01
                    config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
                    
                    
                generated_samples = sample.sample(config,distribution)
                mmd_stats[k][i] = mmd.get_mmd_squared(generated_samples,true_samples).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(generated_samples,true_samples).detach().item()
                
                k+=1
    else:
        mmd_stats = torch.load(os.path.join(folder,'mmd.pt')).to(device=device).to(dtype=torch.float32)
        w2_stats = torch.load(os.path.join(folder,'w2.pt')).to(device=device).to(dtype=torch.float32)
        
        method_names = np.load(os.path.join(folder,'method_names.npy'))
    
    # Save method names and samples
    torch.save(mmd_stats,os.path.join(folder,'mmd.pt'))
    torch.save(w2_stats,os.path.join(folder,'w2.pt'))
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    plt.rcParams.update({'font.size': 14})
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']
    
    for i,method in enumerate(method_names):
        method_label = method[0].upper() + method[1:]
        if method == 'Ground Truth' or method[-2:] != 'MC':
            continue
        print(method)
        ax1.plot(dimensions,mmd_stats[i],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax2.plot(dimensions,w2_stats[i],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    # ax1.set_title('MMD as a function of mode separation')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('MMD')
    ax1.legend(loc='upper left',bbox_to_anchor=(0.6,0.8))
    # ax2.set_title('W2 as a function of mode separation')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('W2')
    ax2.legend(loc='upper left')
    fig.savefig(os.path.join(folder,'dimension_mmd_results.pdf'),bbox_inches='tight')


        