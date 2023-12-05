import torch
import wandb
from math import pi

from utils.integrators import get_integrator
from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers.proximal_sampler as proximal_sampler
import samplers.ula as ula


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

    logdensity, grad_logdensity = dist.log_prob, dist.grad_log_prob
    p0 = lambda x : torch.exp(logdensity(x))
    potential = lambda x :  - logdensity(x)
    gradient = lambda x : p0(x) * grad_logdensity(x)
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

    def get_fourier_estimator(x,tt):
        x = x.unsqueeze(-1)
        scaling = sde.scaling(tt) #e**-t
        var = (scaling * sde.scheduling(tt))**2
        l = config.integration_range
        num_samples = config.sub_intervals_per_dim
        pt = get_push_forward(scaling,p0)
        phase = -l/2.
        dx = l/num_samples
        pts_for_ifft = torch.fft.fftfreq(num_samples,d=dx, device=device)
        pts_for_ifft = torch.fft.fftshift(pts_for_ifft)
        pts = torch.linspace(phase, -phase, num_samples, device=device)
        correct_phase = (pi * torch.arange(0,num_samples, device= device)).unsqueeze(-1)
        correct_phase = torch.view_as_complex(torch.concat([torch.zeros_like(correct_phase,device=device),correct_phase],dim=-1))
        correct_phase = correct_phase.unsqueeze(-1)

        fft_pts = torch.fft.fft(pt(pts),norm='ortho').unsqueeze(-1)
        fft_pts = torch.fft.fftshift(fft_pts)
        # fft_pts = correct_phase * fft_pts
        fft_pts = torch.abs(fft_pts)
        print(fft_pts)
        gaussian_part = torch.exp(-2 * pi **2 * (1-scaling**2) * pts_for_ifft**2).unsqueeze(-1)
        non_osc = (fft_pts * gaussian_part)
        import matplotlib.pyplot as plt
        exact_vals = torch.exp(-2 * pi **2 * scaling ** 2 * pts_for_ifft **2 )

        plt.plot(pts_for_ifft.cpu().numpy(), fft_pts.cpu().numpy())
        plt.plot(pts_for_ifft.cpu().numpy(), exact_vals.cpu().numpy())
        plt.legend(['approximated','real'])
        plt.show()
        def get_density_estimator():
            x_shaped = x.unsqueeze(-1)
            points = pts_for_ifft.unsqueeze(-1)
            exponent = 2. * pi * points * x_shaped
            exponent = torch.view_as_complex(torch.concat([torch.zeros_like(exponent,device=device),exponent],dim=-1))
            osc = torch.exp(exponent).real.unsqueeze(-1)
            complex_part = (osc * non_osc).real
            unnormalized = (l * torch.mean(complex_part, dim=(1,2))).real

            return  unnormalized 
        
        return get_density_estimator()

    if config.score_method == 'p0t' and config.p0t_method == 'rejection':
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dim,device=device),potential)
        # minimizer = torch.tensor([3.,-2.],device=device)
        print(f'Found minimizer {minimizer.cpu().numpy()}')
        
    def get_samplers_based_on_sampling_p0t(x,tt):
        scaling = sde.scaling(tt)
        var = scaling**2 * sde.scheduling(tt)**2

        
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

            # print(num_good_samples.squeeze(-1).cpu().numpy())

            wandb.log({'Average Acc Samples' : torch.mean(num_good_samples).detach().item(), 
                        'Min Acc Samples' : torch.min(num_good_samples).detach().item()})
            num_good_samples[num_good_samples == 0] += 1 # Ask if this is fine
            mean_estimate /= num_good_samples

        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        # import matplotlib.pyplot as plt
        # from . import gmm_score
        # real_log_dens, real_grad = gmm_score.get_gmm_density_at_t(config,sde,tt,device)
        # real_score = real_grad(x)/torch.exp(real_log_dens(x))
        # l = 3
        
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
        # samples_from_p0t = samples_from_p0t[acc_idx].view((-1,dim))
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