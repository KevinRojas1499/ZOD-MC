import torch

import utils.plots
import utils.samplers
import utils.densities
import utils.analytical_score
import utils.sde_utils
import wandb

def get_run_name(config):
    return config.density + "_" + config.sampling_method + "_" + config.convolution_integrator

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name,
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)


def eval(config):
    init_wandb(config)
    # Create files
    ckpt_path = config.work_dir + config.ckpt_path
    # os.makedirs(samples_dir)
    
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    # Get SDE:
    sde = utils.sde_utils.get_sde(config)

    # Load Model
    if config.score_method == 'trained':
        model = torch.load(ckpt_path)
        model.to(device)
    else:
        model = utils.analytical_score.get_score_function(config, sde, device)
    
    # Get Sampler
    sampler = utils.samplers.get_sampler(config,device, sde)

    z_0 = sampler(model)
    if config.dimension == 1:
        utils.plots.histogram(to_numpy(z_0.squeeze(-1)), log_density= utils.densities.get_log_density_fnc(config,device=device)[0])
    elif config.dimension == 2:
        utils.plots.plot_2d_dist(to_numpy(z_0))

def to_numpy(x):
    return x.cpu().detach().numpy()
