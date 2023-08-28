import torch

from utils.integrators import get_integrator
from numpy  import Inf
from utils.densities import get_log_density_fnc
def get_score_function(config, sde):
    """
        The following method returns a method that approximates the score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logdensity = get_log_density_fnc(config)
    p0 = lambda x : torch.exp(logdensity(x))

    def score_gaussian_convolution(x, tt):
        """
            The following method computes the score by making use that:

            p_t = S_# p0 * N(0,1-e^{-2t})
            S(x) = e^-t x 
        """
        def get_convolution(f, g):
            """
                This returns the function f*g
            """
            integrator = get_integrator(config)
            
            def convolution(x):
                def integrand(y):
                    y = y.to(device)
                    return f(y) * g(x-y)
                return integrator(integrand,- config.integration_range, config.integration_range)
            
            return convolution
        
        def get_push_forward(t):
            """
                This returns S_# p0
            """

            def push_forward(x):
                exp = torch.exp(t)
                return p0(exp * x) * exp
            
            return push_forward
        
        t = sde.sigma * tt
        var = 1-torch.exp(-2 * t)
        gaussian = torch.distributions.normal.Normal(0,var ** .5)
        gaussian_density = lambda x : torch.exp(gaussian.log_prob(x))
        grad_gaussian = lambda x : -x * gaussian_density(x) / var

        sp0 = get_push_forward(t)
        
        p_t = get_convolution(sp0, gaussian_density)
        grad_p_t = get_convolution(sp0,grad_gaussian)

        return grad_p_t(x)/p_t(x)
    
    return score_gaussian_convolution