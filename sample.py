import torch
from tqdm import tqdm
import utils.plots
import utils.samplers
import utils.densities
import utils.score_estimators
import utils.metrics
import sde_lib
import samplers.ula as ula


def sample(config, distribution=None):
    # Set up
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device) \
        if distribution is None else distribution
    sde = sde_lib.get_sde(config)
    model = utils.score_estimators.get_score_function(config,distribution,  sde, device)
    
    # Get Sampler
    sampler = utils.samplers.get_sampler(config,device, sde)

    n_batch = config.num_batches
    n_samples = config.sampling_batch_size
    dim = distribution.dim
    samples = torch.zeros((n_batch,n_samples, dim),dtype=torch.float32, device=device)
    pbar = tqdm(range(n_batch),leave=False)
    for i in pbar:
        pbar.set_description(f"Batch {i}/{n_batch}")
        samples[i] = sampler(model)
    samples = samples.view((-1,dim))
    if config.ula_steps > 0:
        samples = samples.to(device=device)
        samples = ula.get_ula_samples(samples,distribution.grad_log_prob,config.ula_step_size,config.ula_steps)
    
    return samples 