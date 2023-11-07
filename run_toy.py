import torch
import wandb
from tqdm import tqdm

import utils.plots
import utils.samplers
import utils.densities
import utils.score_estimators
import utils.sde_utils
import utils.gmm_statistics

def get_run_name(config):
    if config.score_method == 'quotient-estimator':
        return f"SAMPLING {config.density} {config.sde_type} {config.score_method} {config.num_estimator_samples}"
    if config.score_method == 'convolution':
        return f"SAMPLING {config.density} {config.sde_type} {config.score_method} {config.sub_intervals_per_dim}"

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
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    # Get SDE:
    sde = utils.sde_utils.get_sde(config)
    model = utils.score_estimators.get_score_function(config, sde, device)
    
    # Get Sampler
    sampler = utils.samplers.get_sampler(config,device, sde)

    n_batch = config.num_batches
    dim = config.dimension
    samples = torch.zeros((n_batch,config.sampling_batch_size, dim))
    pbar = tqdm(range(n_batch))
    for i in pbar:
        pbar.set_description(f"Batch {i}/{n_batch}")
        samples[i] = sampler(model)
    samples = samples.view((-1,dim))
    print(torch.sum(torch.isnan(samples)))
    # w, error_means = utils.gmm_statistics.summarized_stats(samples)
    # wandb.log({"Error Weights": w, "Error Means": error_means})

    if config.dimension == 1:
        utils.plots.histogram(to_numpy(samples.squeeze(-1)), log_density= utils.densities.get_log_density_fnc(config,device=device)[0])
    elif config.dimension == 2:
        utils.plots.plot_2d_dist(to_numpy(samples))

    wandb.finish()

def to_numpy(x):
    return x.cpu().detach().numpy()
