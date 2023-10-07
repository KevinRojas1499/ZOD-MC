import torch

from utils.integrators import get_integrator
from utils.densities import get_log_density_fnc



def get_score_function(config, sde, device):
    """
        The following method returns a method that approximates the score
    """

    def get_push_forward(scaling, func):
        """
            This returns S_# func
            where S = scaling
        """

        def push_forward(x):
            return func(x/scaling) * (1/scaling)**config.dimension
        
        return push_forward

    logdensity, gradient = get_log_density_fnc(config, device)
    p0 = lambda x : torch.exp(logdensity(x))

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

        if config.dimension == 1:
            gaussian = torch.distributions.normal.Normal(0,var ** .5)
        else:
            gaussian = torch.distributions.MultivariateNormal(torch.zeros(config.dimension,device=device),var * torch.eye(config.dimension,device=device))
        
        def get_grad_gaussian():

            def grad_gaussian(x):
                dens = gaussian_density(x)
                if config.dimension != 1:
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

        if config.mode == 'experiment':
            return p_t(x), grad_p_t(x)
        else:
            return grad_p_t(x)/(p_t(x) + config.eps_stable)


    

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
                noise = torch.randn((config.num_estimator_samples, config.dimension),device=device)
                z = x.unsqueeze(1)
                return torch.mean(func(z - std * noise), dim=1)
            
            return estimator
        
        def get_conv_gradient_estimator(func):

            def estimator(x):
                # x has shape (n,d), noise has shape (k,d)
                # we need to make it (n,k,d) and then average on k
                noise = torch.randn((config.num_estimator_samples, config.dimension),device=device)
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

        if config.mode == 'experiment':
            return p_t(x), grad_p_t(x)
        else:
            return grad_p_t(x)/(p_t(x) + config.eps_stable)

    
    if config.score_method == 'convolution':
        return score_gaussian_convolution
    elif config.score_method == 'quotient-estimator':
        return score_quotient_estimator