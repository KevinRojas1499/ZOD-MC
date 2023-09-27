import torch

from utils.integrators import get_integrator
from utils.densities import get_log_density_fnc


def get_push_forward(t, func):
    """
        This returns S_# func
    """

    def push_forward(x):
        exp = torch.exp(t)
        return func(exp * x) * exp
    
    return push_forward

def get_score_function(config, sde, device):
    """
        The following method returns a method that approximates the score
    """
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
                    y = y.unsqueeze(1).repeat(1, x.shape[0], 1).to(device)

                    return f(x - y) * g(y)
                return integrator(integrand,- l, l)
            
            return convolution
        
        
        def get_integration_limits(t):
            return (1-torch.exp(-2*t))**.5 * (config.integration_range**2 + t)**.5

        t = sde.sigma * tt
        var = 1-torch.exp(-2 * t)
        gaussian = torch.distributions.normal.Normal(0,var ** .5)
        gaussian_density = lambda x : torch.exp(gaussian.log_prob(x))
        grad_gaussian = lambda x : -x * gaussian_density(x) / var

        sp0 = get_push_forward(t, p0)
        limit = get_integration_limits(t)
        p_t = get_convolution(sp0, gaussian_density,limit)
        grad_p_t = get_convolution(sp0,grad_gaussian,limit)

        if config.dimension == 1:
            return grad_p_t(x)/p_t(x)
        else:
            return grad_p_t(x)/p_t(x).unsqueeze(-1)

    

    def score_quotient_estimator(x, tt):
        """
            The following method computes the score by making use that:

            p_t = S_# p0 * N(0,1-e^{-2t})
            S(x) = e^-t x 
            \nabla_x p_t(x) = \nabla p_t / p_t 
                = \E S_# \nabla p0(x - z(1-e^-2t)^.5) / \E S_# p0((x - z(1-e^-2t)^.5))
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

        # return p_t(x), grad_p_t(x)
        return grad_p_t(x)/(p_t(x)+config.eps_stable) 
    
    if config.score_method == 'convolution':
        return score_gaussian_convolution
    elif config.score_method == 'quotient-estimator':
        return score_quotient_estimator