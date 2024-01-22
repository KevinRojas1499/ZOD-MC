import os
import torch
import numpy as np
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.mmd
import sample
import matplotlib.pyplot as plt
import samplers.ula
import samplers.proximal_sampler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_num_methods(config):
    num = len(config.methods_to_run) + len(config.baselines)
    num += 1 if config.eval_mmd else num
    return num     

def eval(config):
    set_seed(12)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device)
    mmd = utils.mmd.MMDLoss()
    eval_mmd = config.eval_mmd
    dim = config.dimension

    # Baseline
    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods = get_num_methods(config)
        
    method_names = [''] * num_methods
    gradient_complexity = config.num_samples_for_rdmc * np.arange(config.min_num_iters_rdmc,
                                                                config.max_num_iters_rdmc,
                                                                step=config.iters_rdmc_step)
    print(gradient_complexity)
    samples_all_methods = torch.zeros((num_methods,len(gradient_complexity), tot_samples,dim),dtype=torch.double, device=device)

    mmd_stats = np.zeros((num_methods, *gradient_complexity.shape),dtype='double')
    k = 0
    if eval_mmd:
        real_samples = distribution.sample(tot_samples)
        method_names[0] = 'Ground Truth'
        for i in range(len(gradient_complexity)):
            samples_all_methods[0][i] = real_samples
        k+=1
    
    folder = os.path.dirname(config.save_folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    if not config.load_from_ckpt:
        for method in config.methods_to_run:
            method_names[k] = method
            for i, gc in enumerate(gradient_complexity):
                if method == 'quotient-estimator':
                    config.score_method = 'quotient-estimator'
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc//config.num_estimator_batches
                    config.sampling_eps = config.sampling_eps_rejec
                elif method == 'rejection':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.sampling_eps = config.sampling_eps_rejec
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc//config.num_estimator_batches
                elif method == 'ula':
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula' 
                    config.sampling_eps = config.sampling_eps_rdmc
                    config.num_estimator_samples = config.num_samples_for_rdmc
                    config.num_sampler_iterations = gc//config.num_estimator_samples
                
                samples_all_methods[k][i] = sample.sample(config)
                if eval_mmd:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
            k+=1
        
        # Baselines
        for baseline in config.baselines:
            prev = 0
            method_names[k] = baseline
            in_cond = torch.randn((tot_samples,dim), device=device)
            for i, gc in enumerate(gradient_complexity):
                if baseline == 'langevin': 
                    samples_all_methods[k][i] = samplers.ula.get_ula_samples(in_cond,
                                                                    distribution.grad_log_prob,
                                                                    config.langevin_step_size,
                                                                    config.disc_steps * (gc - prev) ,
                                                                    display_pbar=False)
                elif baseline == 'proximal':
                    samples_all_methods[k][i] = samplers.proximal_sampler.get_samples(in_cond,
                                                                distribution,
                                                                config.proximal_M,
                                                                config.disc_steps * (gc - prev),
                                                                1,
                                                                device,
                                                                max_grad_complexity = config.disc_steps * (gc - prev)
                                                                ).squeeze(1)
                else:
                    print(f'The baseline method {baseline} has not been implemented yet')
                prev = gc
                if eval_mmd:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
            k+=1    
    
    else:
        samples_all_methods = torch.load(config.samples_ckpt).to(device=device).to(dtype=torch.double)
        method_names = np.load(os.path.join(folder,f'method_names.npy'))
        mmd_stats = np.zeros((len(method_names), *gradient_complexity.shape),dtype='double')
        
        if eval_mmd:
            for k, method in enumerate(method_names):
                if method == 'Ground Truth':
                    k-=1
                    continue
                for i, gc in enumerate(gradient_complexity):
                    if eval_mmd:
                        mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
        
    # Save method names and samples
    save_file = os.path.join(folder,f'samples_{config.density}.pt')
    np.save(os.path.join(folder,f'method_names.npy'), np.array(method_names))
    torch.save(samples_all_methods, save_file)
    

    if dim == 2:
        xlim = [-15,15] if config.density in ['lmm','gmm'] else [-2,2]
        ylim = [-15,15] if config.density in ['lmm','gmm']else [-1,2]
        for i, gc in enumerate(gradient_complexity):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob)
            plt.close(fig)
            fig.savefig(os.path.join(folder,f'Gradient_complexity_{gc}_{config.density}.pdf'), bbox_inches='tight')
        
    if eval_mmd:
        fig, ax = plt.subplots()
        for i, method in enumerate(method_names):
            if method == 'Ground Truth':
                continue
            ax.plot(gradient_complexity,mmd_stats[i],label=method)
        ax.set_title('MMD as a function of Gradient Complexity per Discretization Step')
        ax.set_xlabel('Gradient Complexity')
        ax.set_ylabel('MMD')
        ax.legend()
        fig.savefig(os.path.join(folder,f'mmd_results_{dim}_{config.density}.pdf'))
