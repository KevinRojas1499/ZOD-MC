import torch
from tqdm import tqdm
import utils.plots
import utils.samplers
import utils.densities
import utils.score_estimators
import utils.metrics
import sde_lib
import samplers.ula as ula
import configargparse
from utils.densities import Distribution
def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')
    # Mode
    p.add_argument('--score_method', choices=['p0t','recursive'],default='p0t')
    p.add_argument('--p0t_method', choices=['rejection','ula'],default='rejection')
    p.add_argument('--dimension', type=int)
    
    p.add_argument('--f')
    # All diffusion methods
    p.add_argument('--num_estimator_batches', type=int, default=1) # For rejection
    p.add_argument('--num_estimator_samples', type=int, default=10000) # Per batch for rejection
    
    # Minimizer for ZODMC
    p.add_argument('--max_iters_optimization',type=int, default=50)
    
        
    # RDMC - RSDMC
    p.add_argument('--num_sampler_iterations', type=int) # For langevin
    p.add_argument('--ula_step_size',type=float)
    p.add_argument('--num_recursive_steps',type=int, default=6)
    p.add_argument('--rdmc_initial_condition',choices=['normal','delta'],default='normal')
    
    # Sampling Parameters
    p.add_argument('--sampling_method', choices=['ei','em'])
    p.add_argument('--num_batches', type=int)
    p.add_argument('--sampling_batch_size',type=int)
    p.add_argument('--T', type=float) # early stopping    
    p.add_argument('--sampling_eps', type=float) # early stopping
    p.add_argument('--disc_steps',type=int)
    config = p.parse_args()
    return config

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
    dim = config.dimension
    samples = torch.zeros((n_batch,n_samples, dim),dtype=torch.double, device=device)
    pbar = tqdm(range(n_batch),leave=False)
    for i in pbar:
        pbar.set_description(f"Batch {i}/{n_batch}")
        samples[i] = sampler(model)
    samples = samples.view((-1,dim))
    return samples 

def zodmc(dist : Distribution, 
          num_samples,
          num_batches,
          disc_steps,
          num_rej_samples, 
          num_rej_batches, 
          T=5,
          stopping_time=1e-3,
          max_opt_iters=50):
    config = parse_arguments()
    config.score_method = 'p0t'
    config.p0t_method = 'rejection'
    config.dimension = dist.dim
    
    config.num_estimator_batches = num_rej_batches
    config.num_estimator_samples = num_rej_samples
    
    config.max_iters_optimization = max_opt_iters
    
    config.sampling_method = 'ei'
    config.num_batches = num_batches
    config.sampling_batch_size = num_samples
    
    config.disc_steps = disc_steps
    config.T = T
    config.sampling_eps = stopping_time
    print(config)
    return sample(config,distribution=dist)