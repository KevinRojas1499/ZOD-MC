import torch
import wandb
import utils.plots
import utils.densities
import utils.mmd
import sample
import utils.gmm_utils

def get_run_name(config):
    tot_samples = config.num_estimator_batches * config.num_estimator_samples
    if config.score_method == 'quotient-estimator':
        return f"Sampling {config.density} {config.score_method} {tot_samples}"
    if config.score_method == 'p0t':
        return f'Sampling {config.density} {config.p0t_method} {tot_samples} {config.sampling_method} {config.reuse_samples}'

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=f'{config.wandb_project_name}',
    name= get_run_name(config),
    tags= [config.tags, f'{config.dimension}d',config.density],
    # track hyperparameters and run metadata
    config=config
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
     

def eval(config):
    setup_seed(1)
    
    init_wandb(config)
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device)
    samples = sample.sample(config)
    real_samples=None
    print(torch.sum(torch.isnan(samples)))
    if config.density in ['gmm','lmm']:
        real_samples = distribution.sample(samples.shape[0])
        mmd = utils.mmd.MMDLoss()
        print(mmd.get_mmd_squared(samples,real_samples))
    utils.plots.plot_samples(config,distribution,samples,real_samples)
    wandb.finish()


        