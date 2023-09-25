import torch
from torch.distributions import Normal, MultivariateNormal
import yaml

class MultivariateGaussian():

    # This is a wrapper for Multivariate Normal
    def __init__(self, mean, cov):
        self.dist = MultivariateNormal(mean, cov)
        self.dim = mean.shape[0]
    
    def log_prob(self,x):
        curr_shape = x.shape
        x = x.reshape((-1,self.dim))
        log_prob = self.dist.log_prob(x).reshape(curr_shape)
        return log_prob

def get_log_density_fnc(config, device):
    params = yaml.safe_load(open(config.density_parameters_path))
    def gmm_logdensity_fnc(c,means,variances):
        n = len(c)
        means, variances = torch.tensor(means,device=device, dtype=torch.float64), torch.tensor(variances,device=device, dtype=torch.float64)
        if config.dimension == 1:
            gaussians = [Normal(means[i],variances[i]**2) for i in range(n)]
        else:
            gaussians = [MultivariateGaussian(means[i],variances[i]) for i in range(n)]

        def log_density(x):
            p = 0
            for i in range(n):
                print(x.shape)
                p+= c[i] * torch.exp(gaussians[i].log_prob(x))
            return torch.log(p)
        
        return log_density

    def double_well_density():
        def double_well_log_density(x):
            potential = lambda x :  (x**2 - 1.)**2/2
            return -potential(x)
        
        return double_well_log_density

    if config.density == 'gmm':
        return gmm_logdensity_fnc(params['coeffs'], params['means'], params['variances'])
    elif config.density == 'double-well':
        return double_well_density()
    else:
        print("Density not implemented yet")
        return