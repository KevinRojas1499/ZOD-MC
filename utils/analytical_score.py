import torch

from utils.integrators import get_integrator
from utils.densities import get_log_density_fnc



def get_score_function(config, sde, device):
    """
        The following method returns a method that approximates the score
    """

    def get_push_forward(t, func):
        """
            This returns S_# func
        """

        def push_forward(x):
            exp = torch.exp(t)
            return func(exp * x) * exp**config.dimension
        
        return push_forward

    logdensity, gradient = get_log_density_fnc(config, device)
    p0 = lambda x : torch.exp(logdensity(x))

    def score_gaussian_convolution(x, tt):
        """
            The following method computes the score by making use that:

            p_t = S_# p0 * N(0,1-e^{-2t})
            S(x) = e^-t x 
        """
        def get_convolution(f, g,limit=None):
            """
                This returns the function f*g
            """
            integrator = get_integrator(config)
            
            def convolution(x):
                nonlocal limit
                l = limit if limit is not None else config.integration_range
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
        
        
        def get_integration_limits(t):
            return (1-torch.exp(-2*t))**.5 * (config.integration_range**2 + t)**.5

        t = sde.sigma * tt
        var = 1-torch.exp(-2 * t)
        if config.dimension == 1:
            gaussian = torch.distributions.normal.Normal(0,var ** .5)
        else:
            gaussian = torch.distributions.MultivariateNormal(torch.zeros(config.dimension),var * torch.eye(config.dimension))
        
        def get_grad_gaussian():

            def grad_gaussian(x):
                dens = gaussian_density(x)
                if config.dimension != 1:
                    dens = dens.unsqueeze(-1)
                return -x *  dens/ var
            
            return grad_gaussian

        gaussian_density = lambda x : torch.exp(gaussian.log_prob(x))
        grad_gaussian = get_grad_gaussian()

        sp0 = get_push_forward(t, p0)
        limit = get_integration_limits(t)
        p_t = get_convolution(sp0, gaussian_density,limit)
        grad_p_t = get_convolution(sp0,grad_gaussian,limit)
        
        if config.mode == 'experiment':
            return p_t(x), grad_p_t(x)
        else:
            return grad_p_t(x)/(p_t(x) + config.eps_stable)


    

    def score_quotient_estimator(x, tt):
        """
            The following method computes the score by making use that:

            p_t = S_# p0 * N(0,1-e^{-2t})
            S(x) = e^-t x 
            \nabla_x p_t(x) = \nabla p_t / p_t 
                = \E \nabla S_# p0(x - z(1-e^-2t)^.5) / \E S_# p0((x - z(1-e^-2t)^.5))
        """

        def get_density_estimator(t, func):
            std = (1-torch.exp(-2 * t))**.5

            def estimator(x):
                # x has shape (n,d), noise has shape (k,d)
                # we need to make it (n,k,d) and then average on k
                noise = torch.randn((config.num_estimator_samples, config.dimension))
                z = x.unsqueeze(1)
                return torch.mean(func(z + noise * std), dim=1)
            
            return estimator
        
        def get_conv_gradient_estimator(t, func):
            std = (1-torch.exp(-2 * t))**.5

            def estimator(x):
                # x has shape (n,d), noise has shape (k,d)
                # we need to make it (n,k,d) and then average on k
                noise = torch.randn((config.num_estimator_samples, config.dimension))
                z = x.unsqueeze(1)
                return torch.mean(noise * func(z + noise * std), dim=1)/std 
            
            return estimator
        
        def get_direct_gradient_estimator(t, func):
            standard_estim = get_density_estimator(t,func)
            def estimator(x):
                return standard_estim(x) * torch.exp(t)
            
            return estimator
        t = sde.sigma * tt

        sp0 = get_push_forward(t, p0)
        sgrad = get_push_forward(t, gradient)

        p_t = get_density_estimator(t, sp0)
        if config.gradient_estimator == 'conv':
            grad_p_t = get_conv_gradient_estimator(t,sp0) 
        elif config.gradient_estimator == 'direct':
            grad_p_t = get_direct_gradient_estimator(t,sgrad) 

        if config.mode == 'experiment':
            return p_t(x), grad_p_t(x)
        else:
            return grad_p_t(x)/(p_t(x) + config.eps_stable)

    
    if config.score_method == 'convolution':
        return score_gaussian_convolution
    elif config.score_method == 'quotient-estimator':
        return score_quotient_estimator