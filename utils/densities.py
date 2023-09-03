import torch
from torch.distributions import Normal, MultivariateNormal
import yaml

def get_log_density_fnc(config):
    params = yaml.safe_load(open(config.density_parameters_path))
    def gmm_logdensity_fnc(c,means,variances):
        n = len(c)
        means, variances = torch.tensor(means,dtype=torch.float64), torch.tensor(variances,dtype=torch.float64)
        if config.dimension == 1:
            gaussians = [Normal(means[i],variances[i]**2) for i in range(n)]
        else:
            gaussians = [MultivariateNormal(means[i],variances[i]) for i in range(n)]

        def log_density(x):
            p = 0
            for i in range(n):
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