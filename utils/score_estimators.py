import torch
import wandb
from math import pi

from utils.integrators import get_integrator
from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers.ula as ula
import samplers.metropolis_random_walk as mrw

def get_score_function(config, dist : Distribution, sde, device):
    """
        The following method returns a method that approximates the score
    """
    logdensity, grad_logdensity = dist.log_prob, dist.grad_log_prob
    p0 = lambda x : torch.exp(logdensity(x))
    potential = lambda x :  - logdensity(x)
    dim = config.dimension

    def score_quotient_estimator(x, tt):
        num_iters = config.num_estimator_batches
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        variance_conv = inv_scaling**2 - 1
        num_samples = config.num_estimator_samples
        big_x = x.repeat_interleave(num_samples,dim=0)
        top, dens_mean = 0, 0
        for _ in range(num_iters):
            samples_q = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv**.5
            dens = p0(samples_q).view(-1,num_samples,1)
            samples_q = samples_q.view(-1,num_samples,dim)
            top += torch.mean(samples_q * dens,dim=1)
            dens_mean += torch.mean(dens,dim=1)
            
        mean_estimate = top/dens_mean
        mean_estimate = torch.nan_to_num(mean_estimate)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)

        # from . import gmm_score
        # real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        # real_score = real_grad(x)/torch.exp(real_log_dens(x))
        
        # score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        # print(torch.mean((real_score - score_estimate)**2).cpu().numpy())
        
        return score_estimate
    
    def score_quotient_estimator_2(x, tt):
        num_iters = config.num_estimator_batches
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        var = inv_scaling**2 - 1
        normalizer = (1-scaling**2)**.5
        num_samples = config.num_estimator_samples
        big_x = x.repeat_interleave(num_samples,dim=0)
        lang = - inv_scaling * dist.grad_log_prob(inv_scaling * x)
        big_lang = lang.repeat_interleave(num_samples,dim=0)
        top, w_sum = 0,0
        for _ in range(num_iters):
            center = inv_scaling * big_x + inv_scaling * normalizer**2 * big_lang
            Z = torch.randn_like(big_x)
            samples_q = center + Z * var**.5
            A = normalizer * big_lang + Z
            logp0t = logdensity(samples_q) \
                - .5 * torch.sum(A**2,dim=-1,keepdim=True)
            logq = - .5 * torch.sum((A + normalizer * big_lang)**2,dim=-1,keepdim=True)
            
            w = torch.exp(logp0t - logq).view(-1,num_samples,1)
            w_sum += torch.sum(w,dim=1)
            Z = Z.view(-1,num_samples,dim)
            top += torch.sum(Z * w, dim=1)
        correction = torch.nan_to_num(top/w_sum) #nans are changed to 0
        print(w[:5])
        score_estimate = lang + correction/normalizer

        from . import gmm_score
        real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        real_score = real_grad(x)/torch.exp(real_log_dens(x))
        
        print(torch.mean((real_score - score_estimate)**2).cpu().numpy())
        
        return score_estimate
    
    def importance_estimator(x, tt, q : Distribution = None):
        # q must implement sample()
        num_iters = config.num_estimator_batches
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        variance_conv = inv_scaling**2 - 1
        num_samples = config.num_estimator_samples
        big_x = x.repeat_interleave(num_samples,dim=0)
        mean_estimate = 0
        for _ in range(num_iters):
            if q is not None and tt > 2:
                samples_q = q.sample(big_x.shape[0])
            else:
                samples_q = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv**.5
                
            dens = p0(samples_q).view(-1,num_samples,1)
            samples_q = samples_q.view(-1,num_samples,dim)
            f = samples_q
            w = dens/torch.mean(dens,dim=1,keepdim=True)
            if q is not None and tt > 2:
                norm = torch.sum((samples_q - inv_scaling * big_x.view(-1,num_samples,dim))**2,dim=-1,keepdim=True)
                qq = (2 * pi * variance_conv) ** (-dim/2) * torch.exp( - norm/(2 * variance_conv))
                w *= qq/torch.exp(q.log_prob(samples_q))
                
            mean_estimate += torch.mean(f * w,dim=1)

        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)

        # from . import gmm_score
        # real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        # real_score = real_grad(x)/torch.exp(real_log_dens(x))
        
        # score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        # print(torch.mean((real_score - score_estimate)**2).cpu().numpy())
        
        return score_estimate
    
    def regression_estimator(x, tt):
        num_iters = config.num_estimator_batches
        inv_scaling = 1/sde.scaling(tt)
        variance_conv = inv_scaling**2 - 1
        N = config.num_estimator_samples
        big_x = x.repeat_interleave(N,dim=0)
        for _ in range(num_iters):
            rd_noise = torch.randn_like(big_x)
            z = inv_scaling * big_x + rd_noise * variance_conv**.5
            dens = p0(z).view(-1,N,1)
            z = z.view(-1,N,dim)
            dens_mean = torch.mean(dens,dim=1,keepdim=True)
            w = dens/dens_mean
            f = rd_noise.view(-1,N,dim)/(1 - torch.exp(-2 * tt))**.5
            beta =  torch.mean((w - 1) * f,dim=1,keepdim=True)/torch.mean((w - 1)**2,dim=1,keepdim=True)
        score_estimate = torch.mean((f - beta) * w, dim=1) +  beta.squeeze(1)
        score_estimate_2 = torch.mean(f * dens/dens_mean,dim=1)
        
        from . import gmm_score
        real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        real_score = real_grad(x)/torch.exp(real_log_dens(x))
        
        print(torch.mean((real_score - score_estimate)**2).cpu().numpy(), torch.mean((real_score - score_estimate_2)**2).cpu().numpy())
        return score_estimate

        

    dist.keep_minimizer = False # We don't need minimizers unless we are in the rejection setting
    if config.score_method == 'p0t' and config.p0t_method == 'rejection':
        dist.keep_minimizer = True
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dim,device=device),potential, config.max_iters_optimization)
        dist.log_prob(minimizer) # To make sure we update with the right minimizer
        previous_samples = None
        prev_acc = None
        
        # print(f'Found minimizer {minimizer.cpu().numpy()}')
    
    def optimization_based_sampling(x,tt):
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        L = config.L
        var = 1/(L * dim + 1/(inv_scaling**2-1))

        grad_p0t = lambda x0 : -dist.grad_log_prob(x0) + (x0-inv_scaling * x)/var
        
        w, num_opt_iters = optimizers.nesterovs_minimizer(x,grad_p0t,var,L,1500)
        print((inv_scaling * x)[:2].cpu().numpy(),w[:2].cpu().numpy())
        sigma = 1/var - L 
        mean = (-grad_p0t(w) - L * w +  inv_scaling * x/ var)/sigma
        normal_var = 1/sigma
        num_samples = config.num_estimator_samples
        big_mean = mean.repeat_interleave(num_samples,dim=0)
        big_x = x.repeat_interleave(num_samples,dim=0)
        potential_p0t = lambda x0 : -dist.log_prob(x0) + .5 * torch.sum((x0-inv_scaling * big_x)**2)/var    
        
        z = torch.randn_like(big_mean)
        samples = big_mean + z * normal_var**.5
        w = torch.exp(-potential_p0t(samples))/torch.exp(- .5 * torch.sum(z**2,dim=-1,keepdim=True))
        samples = samples.view(-1,num_samples,dim)
        w = w.view(-1,num_samples,1)
        w_mean = torch.sum(w,dim=1)
        print(w_mean[:5])
        
        mean_estimate = torch.sum(samples * w,dim=1)/w_mean
        mean_estimate = torch.nan_to_num(mean_estimate)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling **2)
        return score_estimate
    
    def get_samplers_based_on_sampling_p0t(x,tt):
        scaling = sde.scaling(tt)
        inv_scaling = 1/scaling
        variance_conv = inv_scaling**2 - 1
        num_samples = config.num_estimator_samples
        score_estimate = torch.zeros_like(x)
        big_x = x.repeat_interleave(num_samples,dim=0)
        potential_0t = lambda x0 : -logdensity(x0) + torch.sum((big_x - scaling * x0)**2,dim=-1,keepdim=True)/(2 * (1 - scaling**2))
        grad_log_prob_0t = lambda x0 : grad_logdensity(x0) + scaling * (big_x - scaling * x0)/(1 - scaling**2)
        grad_pot_0t = lambda x0 : -grad_logdensity(x0) - scaling * (big_x - scaling * x0)/(1 - scaling**2)
        
        if config.p0t_method == 'rejection':
            num_iters = config.num_estimator_batches
            mean_estimate = 0
            num_good_samples = torch.zeros((x.shape[0],1),device=device)
            for _ in range(num_iters):
                samples_from_p0t, acc_idx = rejection_sampler.get_samples(inv_scaling * x, variance_conv,
                                                                                        dist,
                                                                                        num_samples, 
                                                                                        device)
                num_good_samples += torch.sum(acc_idx, dim=(1,2)).unsqueeze(-1).to(torch.double)/dim
                mean_estimate += torch.sum(samples_from_p0t * acc_idx,dim=1)
                
            # nonlocal previous_samples , prev_acc 
            # reuse = config.reuse_samples
            # if previous_samples is not None and reuse != 'no':
            #     if config.reuse_samples == 'mala':
            #         previous_samples, prev_acc_idx  = mrw.mala_iteration(previous_samples, 
            #                                                                 0.1, potential_0t,grad_pot_0t, device)
            #     elif config.reuse_samples == 'mrw':
            #         previous_samples, prev_acc_idx  = mrw.metropolis_random_walk_iteration(previous_samples, 
            #                                                                 0.1, potential_0t, device)
                    
            #     previous_samples = previous_samples.view((-1,num_samples,dim))
            #     prev_acc_idx = prev_acc_idx.view((-1,num_samples,dim))
            #     mean_estimate+= torch.sum(previous_samples * prev_acc_idx * prev_acc,dim=1)
            #     num_good_samples_prev =  torch.sum(prev_acc_idx * prev_acc, dim=(1,2)).unsqueeze(-1).to(torch.double)/dim
            #     num_good_samples += num_good_samples_prev
            #     wandb.log({'Avg New Samples Prev' : torch.mean(num_good_samples_prev).detach().item() })
            # previous_samples = samples_from_p0t.view((-1,dim))
            # prev_acc = acc_idx

            num_good_samples[num_good_samples == 0] += 1 
            mean_estimate /= num_good_samples

            
            # wandb.log({'Average Acc Samples' : torch.mean(num_good_samples).detach().item(),
            #             'Small Num Acc < 10' : len(num_good_samples[num_good_samples <= 10]),
            #             'Min Acc Samples' : torch.min(num_good_samples).detach().item()})
        elif config.p0t_method == 'ula':
            x0 = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv**.5
            samples_from_p0t = ula.get_ula_samples(x0,grad_log_prob_0t,config.ula_step_size,config.num_sampler_iterations)
            samples_from_p0t = samples_from_p0t.view((-1,num_samples,dim))
            
            mean_estimate = torch.mean(samples_from_p0t, dim = 1)
            
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        # import matplotlib.pyplot as plt
        # # from . import gmm_score
        # # real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        # # real_score = real_grad(x)/torch.exp(real_log_dens(x))
        

        # l = 15 if config.density in ['gmm','lmm'] else 3
        
        # fig, (ax1,ax2) = plt.subplots(1,2)
        # nn = 1500
        # pts = torch.linspace(-l, l, nn)
        # xx , yy = torch.meshgrid(pts,pts,indexing='xy')
        # pts = torch.cat((xx.unsqueeze(-1),yy.unsqueeze(-1)),dim=-1).to(device=device)
        # dens = torch.exp(- (potential(pts) + torch.sum((scaling * pts - x)**2,dim=-1, keepdim=True)/(2*variance_conv))).squeeze(-1).cpu()
        # pts = pts.cpu().numpy()

        # ax1.contourf(xx, yy,dens)
        # ax1.scatter(x[:,0].cpu(),x[:,1].cpu(),color='green')
        # ax1.grid()
        # if config.p0t_method == 'rejection':
        #     samples_from_p0t = samples_from_p0t[acc_idx].view((-1,dim))
        # else:
        #     samples_from_p0t = samples_from_p0t.view((-1,dim))
        # sampsx, sampsy = samples_from_p0t[:,0] , samples_from_p0t[:,1]
        # ax2.hist2d(sampsx.cpu().numpy(),sampsy.cpu().numpy(),bins=100,range= [[-l, l], [-l, l]], density=True)
        # ax2.scatter(x[:,0].cpu(),x[:,1].cpu(),color='green')
        # if old_prev is not None:
        #     ax2.scatter(old_prev[:,0].cpu(),old_prev[:,1].cpu(),color='blue')
        # ax2.grid()
        # fig.set_figheight(6)
        # fig.set_figwidth(12)
        # # fig.suptitle(f'Score Error {torch.sum((real_score - score_estimate)**2)**.5 : .4f}', fontsize=16)
        # fig.savefig(f'./score_generated_samples/{tt : .3f}.png')      
        # plt.close()
        
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

        
    if config.score_method == 'quotient-estimator':
        return score_quotient_estimator
    elif config.score_method == 'p0t':
        return get_samplers_based_on_sampling_p0t
    elif config.score_method == 'recursive':
        return get_recursive_langevin