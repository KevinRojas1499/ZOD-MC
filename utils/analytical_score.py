import torch

from utils.integrators import get_integrator
from numpy  import Inf
from utils.densities import get_log_density_fnc
def get_score_function(config, sde, device):
    """
        The following method returns a method that approximates the score
    """
    logdensity = get_log_density_fnc(config, device)
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
                    y = y.unsqueeze(1).repeat(1, x.shape[0], 1)
                    y = y.to(device)

                    return f(x - y) * g(y)
                return integrator(integrand,- l, l)
            
            return convolution
        
        def get_push_forward(t):
            """
                This returns S_# p0
            """

            def push_forward(x):
                exp = torch.exp(t)
                return p0(exp * x) * exp
            
            return push_forward
        
        def get_integration_limits(t):
            return (1-torch.exp(-2*t))**.5 * (config.integration_range**2 + t)**.5

        t = sde.sigma * tt
        var = 1-torch.exp(-2 * t)
        gaussian = torch.distributions.normal.Normal(0,var ** .5)
        gaussian_density = lambda x : torch.exp(gaussian.log_prob(x))
        grad_gaussian = lambda x : -x * gaussian_density(x) / var

        sp0 = get_push_forward(t)
        limit = get_integration_limits(t)
        p_t = get_convolution(sp0, gaussian_density,limit)
        grad_p_t = get_convolution(sp0,grad_gaussian,limit)

        return grad_p_t(x)/p_t(x).unsqueeze(-1)
    
    return score_gaussian_convolution