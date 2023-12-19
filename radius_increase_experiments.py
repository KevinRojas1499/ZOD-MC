import torch
import numpy as np
import wandb
import utils.gmm_utils
import utils.plots
import utils.densities
import utils.mmd
import sample
import matplotlib.pyplot as plt
import samplers.ula
from math import pi, sin , cos

def get_run_name(config):
    return f'Disc_Steps {config.disc_steps} LangStepSize {config.ula_step_size}'

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

def get_gmm_radius(K,R,device):
    sigma = 1
    c = torch.ones(K,device) * sigma
    means = torch.tensor([[cos(2*pi*i/K),sin(2*pi*i/K)] for i in range(K)]
                         ,dtype=torch.double, device=device)
    means = R * means
    variances = torch.tensor([torch.eye(2) * sigma for i in range(K)],
                             dtype=torch.double, device=device)
    return utils.densities.GaussianMixture(c,means,variances)

def eval(config):
    setup_seed(1)    
    init_wandb(config)
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    mmd = utils.mmd.MMDLoss()

    radiuses = np.arange(1,20)
    mmd_rdm = np.zeros_like(radiuses,dtype='double')
    mmd_rej = np.zeros_like(radiuses,dtype='double')
    mmd_lang = np.zeros_like(radiuses,dtype='double')
    
    for i, radius in enumerate(radiuses):
        distribution = get_gmm_radius(6,radius,device)

        # Baseline
        tot_samples = config.num_batches * config.sampling_batch_size
        real_samples = distribution.sample(tot_samples)
    
        # Reverse Diffusion Monte Carlo
        config.p0t_method = 'ula'
        samples_rdm = sample.sample(config)
        
        # Rejection
        config.p0t_method = 'rejection'
        samples_rejection = sample.sample(config)
        
        # Langevin
        samples_langevin = samplers.ula.get_ula_samples(torch.randn_like(samples_rejection),
                                                        distribution.grad_log_prob,
                                                        .01,config.num_sampler_iterations * config.disc_steps)
         
        plot_limit = 20
        fig = utils.plots.plot_all_samples((samples_rejection,samples_rdm,samples_langevin),
                                           ('Ours','Reverse Diffusion Monte Carlo', 'Langevin'),
                                           plot_limit,distribution.log_prob)
        plt.close(fig)
        fig.savefig(f'plots/Radius_{radius}.png', bbox_inches='tight')
        
        mmd_rdm[i] = mmd.get_mmd_squared(samples_rdm,real_samples).detach().item()
        mmd_rej[i] = mmd.get_mmd_squared(samples_rejection,real_samples).detach().item()
        mmd_lang[i] = mmd.get_mmd_squared(samples_langevin,real_samples).detach().item()
        
        
    print(mmd_lang)
    print(mmd_rej)
    print(mmd_rdm)
    # Save MMD Information
    np.savetxt('mmd_results',(radiuses, mmd_rdm,mmd_rej,mmd_lang))
    fig, ax = plt.subplots()
    ax.plot(radiuses,mmd_rdm,label='RDM')
    ax.plot(radiuses,mmd_rej,label='Ours')
    ax.plot(radiuses,mmd_lang,label='LMC')
    ax.set_title('MMD as a function of mode separation')
    ax.set_xlabel('radius')
    ax.set_ylabel('MMD')
    ax.legend()
    fig.savefig('plots/mmd_results.png')
    wandb.log({'MMD Loss':fig})
    wandb.finish()


        