import os
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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_num_methods(config):
    num = len(config.methods_to_run) + len(config.baselines)
    num += 1 if config.eval_mmd else 0
    return num     

def eval(config):
    set_seed(12)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device)
    mmd = utils.metrics.MMDLoss()
    eval_stats = config.eval_mmd
    dim = config.dimension

    # Baseline
    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods = get_num_methods(config)
    method_names = [''] * num_methods
    oracle_complexity = config.num_samples_for_rdmc * np.arange(config.min_num_iters_rdmc,
                                                                config.max_num_iters_rdmc,
                                                                step=config.iters_rdmc_step)
    print(oracle_complexity)
    samples_all_methods = torch.zeros((num_methods,len(oracle_complexity), tot_samples,dim),dtype=torch.double, device=device)

    mmd_stats = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')
    w2_stats = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')

    k = 0
    if eval_stats:
        real_samples = distribution.sample(tot_samples)
        method_names[0] = 'Ground Truth'
        for i in range(len(oracle_complexity)):
            samples_all_methods[0][i] = real_samples
        k+=1
    
    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)
    
    if not config.load_from_ckpt:
        for method in config.methods_to_run:
            method_names[k] = method
            for i, gc in enumerate(oracle_complexity):
                if method == 'ZOD-MC':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.sampling_eps = config.sampling_eps_rejec
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc//config.num_estimator_batches
                elif method == 'RDMC':
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula' 
                    config.sampling_eps = config.sampling_eps_rdmc
                    config.num_estimator_samples = config.num_samples_for_rdmc
                    config.num_sampler_iterations = gc//config.num_estimator_samples
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.num_recursive_steps = 2
                    config.num_estimator_samples = max(1,int(np.exp(np.log(gc)/(2 * config.num_recursive_steps)))) + 1
                    config.num_sampler_iterations = max(1,int(np.exp(np.log(gc)/(2 * config.num_recursive_steps))))
                    
                samples_all_methods[k][i] = sample.sample(config)
                if eval_stats:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                    w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()
                    
            k+=1
        
        # Baselines
        for baseline in config.baselines:
            prev = 0
            method_names[k] = baseline
            in_cond = torch.randn((tot_samples,dim), dtype=torch.double, device=device)
            for i, gc in enumerate(oracle_complexity):
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
                if eval_stats:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                    w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()
            k+=1    
    
    else:
        samples_all_methods = torch.load(config.samples_ckpt).to(device=device).to(dtype=torch.double)
        method_names = np.load(os.path.join(folder,f'method_names.npy'))
        mmd_stats = np.zeros((len(method_names), *oracle_complexity.shape),dtype='double')
        
        if eval_stats:
            for k, method in enumerate(method_names):
                if method == 'Ground Truth':
                    k-=1
                    continue
                for i, gc in enumerate(oracle_complexity):
                    
                    if eval_stats:
                        mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                        w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()
        
    # Save method names and samples
    save_file = os.path.join(folder,f'samples_{config.density}.pt')
    np.save(os.path.join(folder,f'method_names.npy'), np.array(method_names))
    torch.save(samples_all_methods, save_file)
    np.savetxt(os.path.join(folder,'mmd.txt'), mmd_stats)
    np.savetxt(os.path.join(folder,'w2.txt'), w2_stats)

    if dim == 2:
        take_log = config.density not in ['lmm','gmm'] # This is so that we can have nicer level curves for mueller
        rx, ry = -1, 1
        l = 5
        xlim = [-5,13] if config.density in ['lmm','gmm'] else [-5, 9]
        ylim = [-5,13] if config.density in ['lmm','gmm']else [-8,3.5]
        for i, gc in enumerate(oracle_complexity):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob,take_log)
            plt.close(fig)
            fig.savefig(os.path.join(folder,f'complexity_{gc}_{config.density}.pdf'), bbox_inches='tight')
        
    if eval_stats:
        plt.rcParams.update({'font.size': 14})
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        for i, method in enumerate(method_names):
            if method == 'Ground Truth':
                continue
            ax1.plot(oracle_complexity,mmd_stats[i],label=method)
            ax2.plot(oracle_complexity,w2_stats[i],label=method)
        # ax1.set_title('MMD as a function of Oracle Complexity per Score Evaluation')
        ax1.set_xlabel('Oracle Complexity')
        ax1.set_ylabel('MMD')
        ax1.legend(loc='upper left')
        # ax2.set_title('W2 as a function of Oracle Complexity per Score Evaluation')
        ax2.set_xlabel('Oracle Complexity')
        ax2.set_ylabel('W2')
        ax2.legend(loc='upper left')
        fig.savefig(os.path.join(folder,f'mmd_results_{dim}_{config.density}.pdf'),bbox_inches='tight')
