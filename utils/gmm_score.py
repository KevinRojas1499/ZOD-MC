import yaml
import numpy as np

from utils.densities import * 

def get_gmm_density_at_t(config, sde, t, device):
    params = yaml.safe_load(open(config.density_parameters_path))

    c, means, variances = params['coeffs'], np.array(params['means']), params['variances']
    scale = sde.scaling(t)
    mean_t = means * scale.to('cpu').numpy()
    var_t = variances 
    if config.sde_type == 've' or config.sde_type == 'edm':
        for i in range(len(var_t)):
            if config.sde_type == 've':
                var_t[i] = var_t[i] + t
            else:
                var_t[i] = var_t[i] + t**2
    
    logdensity, grad = gmm_logdensity_fnc(c, mean_t, var_t, device)

    return logdensity, grad
