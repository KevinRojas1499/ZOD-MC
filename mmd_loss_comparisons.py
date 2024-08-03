import os
import torch
import numpy as np
import samplers.parallel_tempering
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt
import samplers.ula
import samplers.proximal_sampler
from slips.samplers.smc import smc_algorithm, init_sample_gaussian, init_log_prob_and_grad_gaussian
from slips.samplers.sto_loc import sto_loc_algorithm, sample_y_init
from slips.samplers.alphas import AlphaGeometric
from slips.samplers.mcmc import MCMCScoreEstimator

import time



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
    def target_log_prob_and_grad(y):
        return  distribution.log_prob(y).flatten(),\
            distribution.grad_log_prob(y) #torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

    mmd = utils.metrics.MMDLoss()
    eval_stats = config.eval_mmd
    dim = distribution.dim

    # Baseline
    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods = get_num_methods(config)
    method_names = [''] * num_methods
    oracle_complexity = config.num_samples_for_rdmc * np.arange(config.min_num_iters_rdmc,
                                                                config.max_num_iters_rdmc,
                                                                step=config.iters_rdmc_step)
    print(oracle_complexity)
    samples_all_methods = torch.zeros((num_methods,len(oracle_complexity), tot_samples,dim),dtype=torch.float32, device=device)

    mmd_stats = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')
    w2_stats = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')
    wall_time = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')

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
                start = time.time()

                print(method, gc)
                if method == 'ZOD-MC':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.sampling_eps = config.sampling_eps_rejec
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc//config.num_estimator_batches
                elif method == 'RDMC':
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula' 
                    config.num_estimator_batches = 1
                    config.sampling_eps = config.sampling_eps_rdmc
                    config.num_estimator_samples = config.num_samples_for_rdmc
                    config.num_sampler_iterations = gc//config.num_estimator_samples
                    config.initial_cond_type = 'delta'
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.num_estimator_batches = 1
                    config.num_recursive_steps = 2
                    config.num_estimator_samples = max(1,int(np.exp(np.log(gc)/(2 * config.num_recursive_steps)))) + 1
                    config.num_sampler_iterations = max(1,int(np.exp(np.log(gc)/(2 * config.num_recursive_steps))))
                    
                samples_all_methods[k][i] = sample.sample(config)
                if eval_stats:
                    end = time.time()
                    wall_time[k][i] = end - start
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                    w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()

            k+=1
        # Baselines
        
        sigma = torch.tensor([config.in_cond_sigma],device=device)
        for baseline in config.baselines:
            prev = 0
            method_names[k] = baseline
            in_cond = sigma * torch.randn((tot_samples,dim), dtype=torch.float32, device=device)
            parallel_curr_state = None
            for i, gc in enumerate(oracle_complexity):
                print(baseline, gc)
                start = time.time()
                
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
                elif baseline == 'parallel':
                    num_chains = config.num_chains_parallel
                    num_iters = config.disc_steps * (gc - prev)//(6 * num_chains)
                    betas = torch.linspace(.2,1.,num_chains, dtype=torch.float32,device=device)
                    in_cond = in_cond if i == 0 else parallel_curr_state
                    samples_all_methods[k][i], parallel_curr_state = samplers.parallel_tempering.parallel_tempering(distribution,
                                                                in_cond,betas, num_iters, config.langevin_step_size, device)
                elif baseline == 'ais':
                    
                    num_chains = config.num_chains_parallel
                    n_mcmc_steps = config.disc_steps * gc //(2 + 2 * num_chains)
                    print('Ais' , gc, num_chains, n_mcmc_steps)
                    samples, weights = smc_algorithm(n_particles=tot_samples,
                        target_log_prob=lambda y :distribution.log_prob(y).flatten(),
                        target_log_prob_and_grad=target_log_prob_and_grad,
                        init_sample=lambda n_samples : init_sample_gaussian(n_samples, sigma, dim, device),
                        init_log_prob_and_grad=lambda x : init_log_prob_and_grad_gaussian(x, sigma),                    
                        betas=torch.linspace(0.0, 1.0, num_chains),
                        n_mcmc_steps=n_mcmc_steps,
                        use_ais=True,
                        verbose=True
                    )
                    samples_all_methods[k][i], weights = samples.detach().cpu(), weights.detach().cpu()
                    
                elif baseline == 'smc':
                    
                    num_chains = config.num_chains_parallel
                    n_mcmc_steps = config.disc_steps * gc //(2 + 2 * num_chains)
                    samples_all_methods[k][i] = smc_algorithm(n_particles=tot_samples,
                            target_log_prob=lambda y : distribution.log_prob(y).flatten(),
                            target_log_prob_and_grad=target_log_prob_and_grad,
                            init_sample=lambda n_samples : init_sample_gaussian(n_samples, sigma, dim, device),
                            init_log_prob_and_grad=lambda x : init_log_prob_and_grad_gaussian(x, sigma),                  
                            betas=torch.linspace(0.0, 1.0, num_chains),
                            n_mcmc_steps=n_mcmc_steps,
                            verbose=True
                        ).detach().cpu()
                elif baseline == 'slips':
                    alpha = AlphaGeometric(a=1.0, b=1.0)
                    
                    K = config.disc_steps
                    num_chains = config.num_samples_for_rdmc
                    n_mcmc_steps = gc//num_chains
                    epsilon, epsilon_end, T = 0.35, 6.62e-03, 1.0
                    score_est = MCMCScoreEstimator(
                        step_size=1e-5,
                        n_mcmc_samples=n_mcmc_steps,
                        log_prob_and_grad=target_log_prob_and_grad,
                        n_mcmc_chains=num_chains,
                        keep_mcmc_length=int(0.5 * n_mcmc_steps)
                    )
                    # Sample the initial point with Langevin-within-Langevin
                    y_init = sample_y_init((tot_samples, dim), sigma=sigma, epsilon=epsilon, alpha=alpha, device=device,
                            n_langevin_steps=32, langevin_init=True, score_est=score_est, score_type='mc')
                    # Run the SLIPS algorithm
                    samples_all_methods[k][i] = sto_loc_algorithm(alpha=alpha, y_init=y_init, K=K, T=T, sigma=sigma, score_est=score_est, score_type='mc',
                        epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True, use_snr_discretization=True,
                        verbose=True
                    ).detach().cpu()
                else:
                    print(f'The baseline method {baseline} has not been implemented yet')
                prev = gc
                samples_all_methods[k,i][samples_all_methods[k,i].abs() > 100] = 0. # it helps stabilize for langevin
                in_cond = samples_all_methods[k][i]
                
                if eval_stats:
                    end = time.time()
                    wall_time[k][i] = end - start
                    if baseline in ['langevin','proximal','parallel'] and k>0:
                        wall_time[k][i] += wall_time[k-1][i]
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                    w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()
                    
            k+=1    
    
    else:
        samples_all_methods = torch.load(config.samples_ckpt).to(device=device).to(dtype=torch.float32)
        method_names = np.load(os.path.join(folder,'method_names.npy'))
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
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    torch.save(samples_all_methods, save_file)
    np.savetxt(os.path.join(folder,'mmd.txt'), mmd_stats)
    np.savetxt(os.path.join(folder,'w2.txt'), w2_stats)
    np.savetxt(os.path.join(folder,'time.txt'), wall_time)
    

    if dim == 2:
        take_log = config.density not in ['lmm','gmm'] # This is so that we can have nicer level curves for mueller
        xlim = [-5,13] if config.density in ['lmm','gmm'] else [-5, 9]
        ylim = [-5,13] if config.density in ['lmm','gmm']else [-8,3.5]
        for i, gc in enumerate(oracle_complexity):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob,take_log)
            plt.close(fig)
            fig.savefig(os.path.join(folder,f'complexity_{gc}_{config.density}.png'), bbox_inches='tight')
    else:
        take_log = config.density not in ['lmm','gmm'] # This is so that we can have nicer level curves for mueller
        xlim = [-13,13] if config.density in ['lmm','gmm'] else [-5, 9]
        ylim = [-13,13] if config.density in ['lmm','gmm']else [-8,3.5]
        for i, gc in enumerate(oracle_complexity):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i,:,:],
                                            method_names,
                                            xlim,ylim,None,take_log)
            plt.close(fig)
            fig.savefig(os.path.join(folder,f'complexity_{gc}_{config.density}.pdf'), bbox_inches='tight')
            
    if eval_stats:
        plt.rcParams.update({'font.size': 14})
        
        ls=['--','-.',':']
        markers=['p','*','s','d','h']
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        fig_time, ax_time = plt.subplots(1,1,figsize=(6,6))
        for i, method in enumerate(method_names):
            if method == 'Ground Truth':
                continue
            ax1.plot(oracle_complexity,mmd_stats[i],label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
            ax2.plot(oracle_complexity,w2_stats[i],label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
            ax_time.plot(oracle_complexity,wall_time[i],label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        # ax1.set_title('MMD as a function of Oracle Complexity per Score Evaluation')
        ax1.set_xlabel('Oracle Complexity')
        ax1.set_ylabel('MMD')
        ax1.legend(loc='lower left',bbox_to_anchor=(0.0,0.2))
        # ax2.set_title('W2 as a function of Oracle Complexity per Score Evaluation')
        ax2.set_xlabel('Oracle Complexity')
        ax2.set_ylabel('W2')
        ax2.legend(loc='lower left',bbox_to_anchor=(0.0,0.2))

        
        ax_time.set_yscale('log')
        ax_time.set_xlabel('Gradient Complexity')
        ax_time.set_ylabel('Time (s)')
        ax_time.legend(loc='upper left')

        fig.savefig(os.path.join(folder,f'mmd_results_{dim}_{config.density}.pdf'),bbox_inches='tight')
        fig_time.savefig(os.path.join(folder,f'time_results_{dim}_{config.density}.pdf'),bbox_inches='tight')
