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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
     

def eval(config):
    setup_seed(123)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device)
    mmd = utils.mmd.MMDLoss()
    is_gmm = (config.density == 'gmm')
    eval_mmd = config.eval_mmd
    dim = config.dimension

    # Baseline
    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods = len(config.methods_to_run) + len(config.baselines)
    if eval_mmd:
        real_samples = distribution.sample(tot_samples)
        num_methods+=1
        
    method_names = [''] * num_methods
    gradient_complexity = config.num_samples_for_rdmc * np.arange(config.min_num_iters_rdmc,
                                                                config.max_num_iter_rdmc,
                                                                step=config.iters_rdmc_step)
    print(gradient_complexity)
    samples_all_methods = torch.zeros((num_methods,len(gradient_complexity), tot_samples,dim))
    mmd_stats = np.zeros((num_methods, *gradient_complexity.shape),dtype='double')
    k = 0
    if eval_mmd:
        method_names[0] = 'Ground Truth'
        for i in range(len(gradient_complexity)):
            samples_all_methods[0][i] = real_samples
        k+=1
    
    
    if 'rejection' in config.methods_to_run:
        config.p0t_method = 'rejection'
        config.sampling_eps = config.sampling_eps_rejec
        method_names[k] = 'rejection'
        for i, gc in enumerate(gradient_complexity):
            config.num_estimator_batches = 10
            config.num_estimator_samples = gc//config.num_estimator_batches
            samples_rejection = sample.sample(config)
            samples_all_methods[k][i] = samples_rejection
            if eval_mmd:
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_rejection,real_samples).detach().item()
        k+=1
    if 'ula' in config.methods_to_run:
        method_names[k] = 'rdmc'
        config.p0t_method = 'ula' 
        config.sampling_eps = config.sampling_eps_rdmc
        config.num_estimator_samples = config.num_samples_for_rdmc
        for i, gc in enumerate(gradient_complexity):
            config.num_sampler_iterations = gc//config.num_estimator_samples
            config.sampling_eps = config.sampling_eps_rdmc
            samples_rdm = sample.sample(config)
            samples_all_methods[k][i] = samples_rdm
            if eval_mmd:
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_rdm,real_samples).detach().item()
        k+=1
    
    # Baselines
    for baseline in config.baselines:
        if baseline == 'langevin': 
            method_names[k] = 'langevin'
            for i, gc in enumerate(gradient_complexity):
                samples_langevin = samplers.ula.get_ula_samples(torch.randn_like(samples_rejection),
                                                                distribution.grad_log_prob,
                                                                config.langevin_step_size,
                                                                gc * config.disc_steps,
                                                                display_pbar=False)
                samples_all_methods[k][i] = samples_langevin
                if eval_mmd:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_langevin,real_samples).detach().item()
        elif baseline == 'proximal':
            method_names[k] = 'proximal'
            for i, gc in enumerate(gradient_complexity):
                samples_proximal = samplers.proximal_sampler(torch.randn_like(samples_rejection),
                                                            config.proximal_eta,
                                                            distribution,
                                                            config.proximal_M,
                                                            config.proximal_num_iters,
                                                            1,
                                                            device).unsqueeze(1)
                samples_all_methods[k][i] = samples_proximal
                if eval_mmd:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_proximal,real_samples).detach().item()
        else:
            print('The baseline method has not been implemented yet')
        k+=1    
    
    
    
    if dim == 2:
        xlim = [-15,15] if is_gmm else [-2,2]
        ylim = [-15,15] if is_gmm else [-1,2]
        for i in range(num_methods):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i],
                                            method_names,
                                            xlim,ylim,distribution.log_prob)
            plt.close(fig)
            fig.savefig(f'plots/Gradient_complexity_{gradient_complexity[i]}_{config.density}.png', bbox_inches='tight')
        
    if eval_mmd:   
        np.savetxt('mmd_results',(gradient_complexity, mmd_stats))
        fig, ax = plt.subplots()
        for i, method in enumerate(method_names):
            if method == 'Ground Truth':
                continue
            ax.plot(gradient_complexity,mmd_stats[i],label=method)
        ax.set_title('Gradient Complexity per Discretization Step')
        ax.set_xlabel('Gradient Complexity')
        ax.set_ylabel('MMD')
        ax.legend()
        fig.savefig(f'plots/mmd_results_{dim}_{config.density}.png')