import torch
import wandb
from math import pi

from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers.ula as ula

def get_score_function(config, dist : Distribution, sde, device):
    """
        The following method returns a method that approximates the score
    """
    logdensity, grad_logdensity = dist.log_prob, dist.grad_log_prob
    p0 = lambda x : torch.exp(logdensity(x))
    potential = lambda x :  - logdensity(x)
    dim = dist.dim
    dist.keep_minimizer = False # We don't need minimizers unless we are in the rejection setting
    if config.score_method == 'p0t' and config.p0t_method == 'rejection':
        dist.keep_minimizer = True
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dim,device=device),potential, config.max_iters_optimization)
        dist.log_prob(minimizer) # To make sure we update with the right minimizer
        # print(f'Found minimizer {minimizer.cpu().numpy()}')

    def get_samplers_based_on_sampling_p0t(x,tt):
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        variance_conv = inv_scaling**2 - 1
        num_samples = config.num_estimator_samples
        score_estimate = torch.zeros_like(x)
        big_x = x.repeat_interleave(num_samples,dim=0)
        grad_log_prob_0t = lambda x0 : grad_logdensity(x0) + scaling * (big_x - scaling * x0)/(1 - scaling**2)
        
        if config.p0t_method == 'rejection':
            num_iters = config.num_estimator_batches
            mean_estimate = 0
            num_good_samples = torch.zeros((x.shape[0],1),device=device)
            for _ in range(num_iters):
                samples_from_p0t, acc_idx = rejection_sampler.get_samples(inv_scaling * x, variance_conv,
                                                                                        dist,
                                                                                        num_samples, 
                                                                                        device)
                num_good_samples += torch.sum(acc_idx, dim=(1,2)).unsqueeze(-1).to(torch.float32)/dim
                mean_estimate += torch.sum(samples_from_p0t * acc_idx,dim=1)
            num_good_samples[num_good_samples == 0] += 1 
            mean_estimate /= num_good_samples
        elif config.p0t_method == 'ula':
            x0 = big_x
            if config.rdmc_initial_condition == 'normal':
                x0 = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv**.5
            
            samples_from_p0t = ula.get_ula_samples(x0,grad_log_prob_0t,config.ula_step_size,config.num_sampler_iterations)
            samples_from_p0t = samples_from_p0t.view((-1,num_samples,dim))
            
            mean_estimate = torch.mean(samples_from_p0t, dim = 1)
            
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)

        return score_estimate
    
    
    def get_recursive_langevin(x,tt,k=config.num_recursive_steps):
        if k == 0 or tt < .2:
            return grad_logdensity(x)
        
        num_samples = config.num_estimator_samples
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        h = config.ula_step_size      

        big_x = x.repeat_interleave(num_samples,dim=0) 
        x0 = big_x.detach().clone()    
        # x0 = inv_scaling * x0 + torch.randn_like(x0) * (inv_scaling**2 -1)  # q0 initialization
        for _ in range(config.num_sampler_iterations):
            score = get_recursive_langevin(x0, (k-1) * tt/k,k-1) + scaling * (big_x - scaling * x0)/(1-scaling**2)
            x0 = x0 + h * score + (2*h)**.5 * torch.randn_like(x0)
        x0 = x0.view((-1,num_samples,dim))
        mean_estimate = x0.mean(dim=1)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate

        
    if config.score_method == 'p0t':
        return get_samplers_based_on_sampling_p0t
    elif config.score_method == 'recursive':
        return get_recursive_langevin