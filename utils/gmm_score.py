import yaml
import numpy as np

from utils.densities import * 

def get_gmm_density_at_t(config, sde, t, device):
    params = yaml.safe_load(open(config.density_parameters_path))

    c, means, variances = params['coeffs'], np.array(params['means']), np.array(params['variances'])
    scale = sde.scaling(t).cpu().numpy()
    mean_t = means * scale
    var_t = variances * scale**2 + (1-scale**2) * np.eye(2)
    logdensity, grad = gmm_logdensity_fnc(c, mean_t, var_t, device)

    return logdensity, grad
