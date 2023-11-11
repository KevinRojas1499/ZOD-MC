import yaml
import torch
import utils.score_estimators
import utils.integrators
import utils.sde_utils
from utils.densities import * 
from utils.plots import *
import plotly.graph_objects as go
from tqdm import tqdm

def get_gmm_density_at_t(config, sde, t):
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
        
    return c,mean_t,var_t

def to_numpy(x):
    return x.cpu().numpy()

def run_fourier_experiments(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = utils.sde_utils.get_sde(config)
    t = torch.tensor([0.01],device=device)

    c, mean_t, var_t = get_gmm_density_at_t(config, sde, t)
    p0, grad = gmm_logdensity_fnc(c, mean_t, var_t, device)
    est_density = utils.score_estimators.get_score_function(config,sde,device)



    x = torch.linspace(-10,10,200,device=device)
    est_dens_pts = est_density(x,t)
    plt.plot(to_numpy(x), to_numpy(torch.exp(p0(x))))
    plt.plot(to_numpy(x), to_numpy(est_dens_pts),color='red')
    plt.legend(['real','approx'])
    plt.show()