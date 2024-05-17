import os
import yaml
import torch
import numpy as np
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt
import samplers.ula
import samplers.proximal_sampler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def to_tensor_type(x, device):
    return torch.tensor(x,device=device, dtype=torch.float32)  

def get_gmm_radius(config,R,device):
    params = yaml.safe_load(open(config.density_parameters_path))
    c = to_tensor_type(params['coeffs'],device)
    means = to_tensor_type(params['means'],device)
    variances = to_tensor_type(params['variances'],device)
    means = R * means
    gaussians = [utils.densities.MultivariateGaussian(means[i],variances[i]) for i in range(c.shape[0])]
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
    radiuses = np.arange(1,10,step=2)
    num_rad = len(radiuses)
    mmd_stats = np.zeros([num_methods, num_rad],dtype='double')
    w2_stats = np.zeros([num_methods, num_rad],dtype='double')
    
    samples_all = torch.zeros([num_methods, num_rad,tot_samples, config.dimension],device=device,dtype=torch.float32)
    
    
    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)

    if not config.load_from_ckpt:
        for i, r in enumerate(radiuses):
            distribution = get_gmm_radius(config,r,device)
            # Baseline
            samples_all[0][i] = distribution.sample(tot_samples)
            k = 1
            for method in config.methods_to_run:
                if method == 'ZOD-MC':
                    # Rejection
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.T = 10
                    config.num_estimator_batches = 1
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
                    config.num_sampler_iterations = 3
                    config.ula_step_size = 0.01
                    config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
                    
                    
                samples_all[k][i] = sample.sample(config,distribution)
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(samples_all[k][i],samples_all[0][i]).detach().item()
                
                k+=1
                
            for method in config.baselines:
                in_cond = torch.randn_like(samples_all[0][i])
                if method == 'langevin':
                    # Langevin
                    distribution.keep_minimizer = False
                    ula_step_size = 0.01
                    num_steps_lang = 50000 
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
                elif method == 'parallel':
                    num_chains = config.num_chains_parallel
                    num_iters = 50000
                    betas = torch.linspace(.2,1.,num_chains, dtype=torch.float32,device=device)
                    samples_all[k][i] = samplers.parallel_tempering.parallel_tempering(distribution,
                                                                in_cond,betas, num_iters, config.langevin_step_size, device)
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(samples_all[k][i],samples_all[0][i]).detach().item()
                k+=1
            xlim = [-4, 8*r + 4]
            ylim = [-4, 8*r + 4]
            fig = utils.plots.plot_all_samples(samples_all[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob)
            fig.savefig(os.path.join(folder,f'radius_{r}.pdf'), bbox_inches='tight')
            plt.close(fig)
    else:
        samples_all = torch.load(config.samples_ckpt).to(device=device).to(dtype=torch.float32)
        method_names = np.load(os.path.join(folder,f'method_names.npy'))
        
        for i, r in enumerate(radiuses):
            for k, method in enumerate(method_names):
                if method == 'Ground Truth':
                    k-=1
                    continue
                distribution = get_gmm_radius(config,r,device)
                
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(samples_all[k][i],samples_all[0][i]).detach().item()
                print(f'{method} {r} {torch.sum((samples_all[k][i][:,0] < 30))} {torch.sum((samples_all[k][i][:,1] < 30))}')
                xlim = [-4, 8*r + 4]
                ylim = [-4, 8*r + 4]
            fig = utils.plots.plot_all_samples(samples_all[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob)
            fig.savefig(os.path.join(folder,f'radius_{r}.pdf'), bbox_inches='tight')
            plt.close(fig)
    
    # Save method names and samples
    save_file = os.path.join(folder,f'samples_{config.density}.pt')
    np.save(os.path.join(folder,f'method_names.npy'), np.array(method_names))
    torch.save(samples_all, save_file)
    plt.rcParams.update({'font.size': 14})
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']
    
    for i,method in enumerate(method_names):
        method_label = method[0].upper() + method[1:]
        if method == 'Ground Truth':
            continue
        print(method)
        ax1.plot(radiuses,mmd_stats[i],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax2.plot(radiuses,w2_stats[i],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    # ax1.set_title('MMD as a function of mode separation')
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('MMD')
    ax1.legend(loc='upper left',bbox_to_anchor=(0.6,0.8))
    # ax2.set_title('W2 as a function of mode separation')
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('W2')
    ax2.legend(loc='upper left')
    fig.savefig(os.path.join(folder,f'radius_mmd_results.pdf'),bbox_inches='tight')


        