import yaml
import torch

from utils.densities import * 

def get_gmm_density_at_t(config, sde, t, device):
    def to_tensor_type(x):
        return torch.tensor(x,device=device, dtype=torch.double)    
    params = yaml.safe_load(open(config.density_parameters_path))

    c = to_tensor_type(params['coeffs'])
    means = to_tensor_type(params['means'])
    variances = to_tensor_type(params['variances'])
    scale = sde.scaling(t)
    mean_t = means * scale
    var_t = variances * scale**2 + (1-scale**2) * torch.eye(config.dimension,device=device)
    dist = GaussianMixture(c, mean_t, var_t)

    return dist.log_prob, dist.gradient
