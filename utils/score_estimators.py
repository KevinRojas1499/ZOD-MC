import torch
import wandb
from math import pi

from utils.integrators import get_integrator
from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers.proximal_sampler as proximal_sampler
import samplers.ula as ula
import samplers.metropolis_random_walk as met_rand_walk


def get_score_function(config, dist : Distribution, sde, device):
    """
        The following method returns a method that approximates the score
    """

    def get_push_forward(scaling, func):
        """
            This returns S_# func
            where S = scaling
        """

        def push_forward(x):
            return func(x/scaling) * (1/scaling)**dim
        
        return push_forward

    logdensity, grad_logdensity, gradient = dist.log_prob, dist.grad_log_prob, dist.gradient
    p0 = lambda x : torch.exp(logdensity(x))
    potential = lambda x :  - logdensity(x)
    dim = config.dimension
    
    def score_gaussian_convolution(x, tt):
        """
            The following method computes the score by making use that:

            p_t = \int S_# p0(x - s\sigma y) * N(y; 0,1) dy
            Where S(x) is the scaling of the sde at time t
        """
        def get_convolution(f, g):
            """
                This returns the function f*g
            """
            integrator = get_integrator(config)
            
            def convolution(x):
                l = config.integration_range
                def integrand(y):
                    # The shape of y is (k,d) where k is the number of eval points
                    # We reshape it to (k,n,d) where n is the number of points in x
                    nonlocal x
                    y = y.unsqueeze(1).to(device)
                    x_shape = x.unsqueeze(0)
                    f_vals = f(x_shape-y)
                    shape_g = g(y)
                    if len(f_vals.shape) != len(shape_g.shape):
                        # This can happen because we have different shapes for the gradient vs the density
                        shape_g = shape_g.unsqueeze(-1)
                    return f_vals * shape_g
                return integrator(integrand,- l, l)
            
            return convolution

        scaling = sde.scaling(tt)
        var = scaling**2 * sde.scheduling(tt)**2

        if dim == 1:
            gaussian = torch.distributions.normal.Normal(0,var ** .5)
        else:
            gaussian = torch.distributions.MultivariateNormal(torch.zeros(dim,device=device),var * torch.eye(dim,device=device))
        
        def get_grad_gaussian():

            def grad_gaussian(x):
                dens = gaussian_density(x)
                if dim != 1:
                    dens = dens.unsqueeze(-1)
                return -x *  dens / var
            
            return grad_gaussian

        gaussian_density = lambda x : torch.exp(gaussian.log_prob(x))
        grad_gaussian = get_grad_gaussian()

        sp0 = get_push_forward(scaling, p0)
        p_t = get_convolution(sp0, gaussian_density)
        if config.gradient_estimator == 'conv':
            grad_p_t = get_convolution(sp0,grad_gaussian)
        elif config.gradient_estimator == 'direct':
            grad_p_t = get_convolution(gradient,gaussian_density)

        if config.mode == 'sample':
            return grad_p_t(x)/(p_t(x) + config.eps_stable)
        else :
            return p_t(x), grad_p_t(x)


    

    def score_quotient_estimator(x, tt):
        """
            The following method computes the score by making use that:

            p_t = E_z [ p_#t (x- s \sigma z)] z \sim N(0,1)
            p_#t = (s^-1)_# p0
        """
        scaling = sde.scaling(tt)
        std = scaling * sde.scheduling(tt)

        def get_density_estimator(func):

            def estimator(x):
                # x has shape (n,d), noise has shape (k,d)
                # we need to make it (n,k,d) and then average on k
                noise = torch.randn((config.num_estimator_samples, dim),device=device)
                z = x.unsqueeze(1)
                return torch.mean(func(z - std * noise), dim=1)
            
            return estimator
        
        def get_conv_gradient_estimator(func):

            def estimator(x):
                # x has shape (n,d), noise has shape (k,d)
                # we need to make it (n,k,d) and then average on k
                noise = torch.randn((config.num_estimator_samples, dim),device=device)
                z = x.unsqueeze(1)
                return -torch.mean(noise * func(z - std * noise), dim=1) / std 
            
            return estimator
        
        def get_direct_gradient_estimator(func):
            standard_estim = get_density_estimator(func)
            def estimator(x):
                return standard_estim(x) / std
            
            return estimator
        


        sp0 = get_push_forward(scaling, p0)
        sgrad = get_push_forward(scaling, gradient)

        p_t = get_density_estimator(sp0)
        if config.gradient_estimator == 'conv':
            grad_p_t = get_conv_gradient_estimator(sp0) 
        elif config.gradient_estimator == 'direct':
            grad_p_t = get_direct_gradient_estimator(sgrad) 

        if config.mode == 'sample':
            return grad_p_t(x)/(p_t(x) + config.eps_stable)
        else :
            return p_t(x), grad_p_t(x)

    
    if config.score_method == 'fourier':
        lu = config.integration_range
        m = config.sub_intervals_per_dim
        dx = lu/m
        idx = torch.arange(-m/2,m/2,device=device)
        pts = idx * dx
        pts = torch.fft.ifftshift(pts)
        fft = torch.fft.fft(p0(pts)) * dx     
        dw = 1/lu
        
    def get_fourier_estimator(x,tt):
        x = x.unsqueeze(-1)
        scaling = sde.scaling(tt) #e**-t
        inv_sc = 1/scaling #e**t

        density = 0
        gradient = 0
        grad_exp = torch.view_as_complex(torch.tensor([0,2 * pi * dw],device=device))
        exp = torch.view_as_complex(torch.cat((torch.zeros_like(x),
                                        2 * pi * dw * x),dim=-1))
        for k in idx:
            k = k.long().item()
            next_term = fft[k] \
                * torch.exp(-2 * (inv_sc**2 - 1) * pi **2 * dw **2 * k**2) \
                * torch.exp(inv_sc * k * exp)
            density = density + next_term
            gradient = gradient + next_term * grad_exp * k
        density = inv_sc * dw * density
        gradient = inv_sc**2 * dw * gradient
        
        if config.mode == 'sample':
            return gradient.real/density.real
        else:
            return torch.abs(density.real), gradient.real

    if config.score_method == 'p0t' and config.p0t_method == 'rejection':
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dim,device=device),potential)
        print(f'Found minimizer {minimizer.cpu().numpy()}')
        
    def get_samplers_based_on_sampling_p0t(x,tt):
        scaling = sde.scaling(tt)
        
        variance_conv = 1/scaling**2 - 1
        num_samples = config.num_estimator_samples
        
        score_estimate = torch.zeros_like(x)
        y = x/scaling
        y_for_sampling = y.repeat_interleave(num_samples,dim=0)
        potential_p0t = lambda x : -logdensity(x) + torch.sum((x-y_for_sampling)**2,dim=1, keepdim=True)/(2*variance_conv)
        gradient_p0t = lambda x : - grad_logdensity(x) + (x-y_for_sampling)/variance_conv
                
        if config.p0t_method == 'proximal':
            M = config.proximal_M
            eta = 1/(M*dim)
            samples_from_p0t, average_rejection_iters = proximal_sampler.get_samples(x, eta,potential_p0t,
                            gradient_p0t,M, 
                            config.num_proximal_iterations, 
                            num_samples, device)
        if config.p0t_method == 'rejection':
            num_iters = config.num_estimator_batches
            mean_estimate = 0
            num_good_samples = torch.zeros((x.shape[0],1),device=device)
            k = 0
            while k < num_iters:
                if torch.min(num_good_samples).detach().item() >= 5:
                    break
                samples_from_p0t, acc_idx = rejection_sampler.get_samples(y, variance_conv,
                                                                                        potential,
                                                                                        num_samples, 
                                                                                        device,
                                                                                        minimizer=minimizer)
                num_good_samples += torch.sum(acc_idx, dim=(1,2)).unsqueeze(-1).to(torch.float32)/dim
                mean_estimate += torch.sum(samples_from_p0t * acc_idx,dim=1)
                k+=1
            # print(len(num_good_samples[num_good_samples == 0]))

            num_good_samples[num_good_samples == 0] += 1 # Ask if this is fine
            mean_estimate /= num_good_samples
        elif config.p0t_method == 'random_walk':
            samples_from_p0t = met_rand_walk.get_samples(x,1/scaling, 
                                                         variance_conv,
                                                         potential,num_samples, 
                                                         config.num_sampler_iterations, 
                                                         device)
            mean_estimate = torch.mean(samples_from_p0t, dim=1)
            
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        # import matplotlib.pyplot as plt
        # from . import gmm_score
        # real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        # real_score = real_grad(x)/torch.exp(real_log_dens(x))
        
        wandb.log({'Average Acc Samples' : torch.mean(num_good_samples).detach().item(),
            'Small Num Acc < 10' : len(num_good_samples[num_good_samples <= 10]),
            'Min Acc Samples' : torch.min(num_good_samples).detach().item()})
        # l = 15
        
        # fig, (ax1,ax2) = plt.subplots(1,2)
        # nn = 1500
        # pts = torch.linspace(-l, l, nn)
        # xx , yy = torch.meshgrid(pts,pts,indexing='xy')
        # pts = torch.cat((xx.unsqueeze(-1),yy.unsqueeze(-1)),dim=-1).to(device=device)
        # dens = torch.exp(- (potential(pts) + torch.sum((pts - y)**2,dim=-1, keepdim=True)/(2*variance_conv))).squeeze(-1).cpu()
        # pts = pts.cpu().numpy()

        # ax1.contourf(xx, yy,dens)
        # ax1.scatter(x[:,0].cpu(),x[:,1].cpu(),color='green')
        # ax1.grid()
        # # samples_from_p0t = samples_from_p0t[acc_idx].view((-1,dim))
        # samples_from_p0t = samples_from_p0t.view((-1,dim))
        # sampsx, sampsy = samples_from_p0t[:,0] , samples_from_p0t[:,1]
        # ax2.hist2d(sampsx.cpu().numpy(),sampsy.cpu().numpy(),bins=100,range= [[-l, l], [-l, l]], density=True)
        # ax2.scatter(x[:,0].cpu(),x[:,1].cpu(),color='green')
        # ax2.grid()
        # fig.set_figheight(6)
        # fig.set_figwidth(12)
        # fig.suptitle(f'Score Error {torch.sum((real_score - score_estimate)**2)**.5 : .4f}', fontsize=16)
        # fig.savefig(f'./score_generated_samples/{tt : .3f}.png')      
        # plt.close()
        

        if config.mode == 'sample' or config.mode == 'generation-experiments':
            return score_estimate
        else:
            return score_estimate, average_rejection_iters
    
    if config.score_method == 'convolution':
        return score_gaussian_convolution
    elif config.score_method == 'quotient-estimator':
        return score_quotient_estimator
    elif config.score_method == 'fourier':
        return get_fourier_estimator
    elif config.score_method == 'p0t':
        return get_samplers_based_on_sampling_p0t