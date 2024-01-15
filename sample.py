import torch
from tqdm import tqdm
import utils.plots
import utils.samplers
import utils.densities
import utils.score_estimators
import utils.mmd
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
    dim = config.dimension
    samples = torch.zeros((n_batch,n_samples, dim),dtype=torch.double, device=device)
    pbar = tqdm(range(n_batch),leave=False)
    for i in pbar:
        pbar.set_description(f"Batch {i}/{n_batch}")
        samples[i] = sampler(model)
    samples = samples.view((-1,dim))
    if config.ula_steps > 0:
        samples = samples.to(device=device)
        samples = ula.get_ula_samples(samples,distribution.grad_log_prob,config.ula_step_size,config.ula_steps)
    
    return samples 

def adaptive_sampling(config, distribution=None):
    # Set up
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device) \
        if distribution is None else distribution
    sde = sde_lib.get_sde(config)
    model = utils.score_estimators.get_score_function(config,distribution,  sde, device)
    sampler = utils.samplers.get_sampler(config,device, sde)
    n_batch = config.num_batches
    n_samples = config.sampling_batch_size
    dim = config.dimension
    
    samples = torch.zeros((n_batch,n_samples, dim),dtype=torch.double, device=device)
    
    sigma = .2
    new_model = None
    for i in range(4):
        # Get Sampler

        pbar = tqdm(range(n_batch),leave=False)
        for i in pbar:
            pbar.set_description(f"Batch {i}/{n_batch}")
            ss= sampler(model) if new_model == None else sampler(new_model)
            print(ss.shape)
            samples[i] = ss
        samples = samples.view((-1,dim)).to(device=device)
        gaussians = [utils.densities.MultivariateGaussian(samples[i],sigma * torch.eye(dim,device=device,dtype=torch.double)) for i in range(n_samples)]
        c = torch.ones(n_samples,device=device,dtype=torch.double)/n_samples
        q = utils.densities.MixtureDistribution(c,gaussians)
        new_model = lambda x,t : model(x,t,q)
        samples = samples.view((-1,n_samples,dim)).to(device=device)
    samples = samples.view((-1,dim)).to(device=device)
        
    return samples