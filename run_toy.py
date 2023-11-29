import torch
import wandb
from tqdm import tqdm
import numpy as np
import random
import utils.plots
import utils.samplers
import utils.densities
import utils.score_estimators
import utils.sde_utils
import utils.gmm_utils as gmm_utils
import samplers.ula as ula

def get_run_name(config):
    if config.score_method == 'quotient-estimator':
        return f"SAMPLING {config.density} {config.sde_type} {config.score_method} {config.num_estimator_samples}"
    if config.score_method == 'convolution':
        return f"SAMPLING {config.density} {config.sde_type} {config.score_method} {config.sub_intervals_per_dim}"
    if config.score_method == 'p0t':
        return f'Sampling {config.density} {config.p0t_method} {config.num_estimator_samples} {config.sampling_method}'

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name,
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
     

def eval(config):
    setup_seed(1)
    
    init_wandb(config)
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    # Get SDE:
    distribution = utils.densities.get_distribution(config,device)
    sde = utils.sde_utils.get_sde(config)
    model = utils.score_estimators.get_score_function(config,distribution,  sde, device)
    
    # Get Sampler
    sampler = utils.samplers.get_sampler(config,device, sde)

    n_batch = config.num_batches
    n_samples = config.sampling_batch_size
    dim = config.dimension
    samples = torch.zeros((n_batch,n_samples, dim))
    pbar = tqdm(range(n_batch))
    for i in pbar:
        pbar.set_description(f"Batch {i}/{n_batch}")
        samples[i] = sampler(model)
    samples = samples.view((-1,dim))
    print(torch.sum(torch.isnan(samples)))
    # w, error_means = utils.gmm_utils.summarized_stats(samples)
    # wandb.log({"Error Weights": w, "Error Means": error_means})

    plot_samples(config, distribution, n_batch, n_samples, samples)
    if config.ula_steps > 0:
        samples = samples.to(device=device)
        samples = ula.get_ula_samples(samples,distribution.grad_log_prob,0.001,config.ula_steps)
        plot_samples(config, distribution, n_batch, n_samples, samples)
        
    wandb.finish()

def plot_samples(config, distribution, n_batch, n_samples, samples):
    if config.dimension == 1:
        utils.plots.histogram(to_numpy(samples.squeeze(-1)), log_density=distribution.log_prob)
    elif config.dimension == 2:
        if config.density == 'gmm':
            real_samples = gmm_utils.sample_from_gmm(config,n_samples * n_batch)
            utils.plots.plot_2d_dist(to_numpy(samples),to_numpy(real_samples))
        else:
            utils.plots.plot_2d_dist_with_contour(to_numpy(samples),distribution.log_prob)

def to_numpy(x):
    return x.cpu().detach().numpy()
