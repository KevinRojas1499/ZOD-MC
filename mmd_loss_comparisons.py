import torch
import numpy as np
import wandb
import utils.plots
import utils.densities
import utils.mmd
import sample
import matplotlib.pyplot as plt
import samplers.ula

def get_run_name(config):
    if config.score_method == 'quotient-estimator':
        return f"SAMPLING {config.density} {config.sde_type} {config.score_method} {config.num_estimator_samples}"
    if config.score_method == 'convolution':
        return f"SAMPLING {config.density} {config.sde_type} {config.score_method} {config.sub_intervals_per_dim}"
    if config.score_method == 'p0t':
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
     

def eval(config):
    setup_seed(1)    
    init_wandb(config)
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device)
    mmd = utils.mmd.MMDLoss()
    
    # Baseline
    tot_samples = config.num_batches * config.sampling_batch_size
    real_samples = distribution.sample(tot_samples)

    gradient_complexity = 10 * np.arange(1,101,step=19)
    mmd_rdm = np.zeros_like(gradient_complexity,dtype='double')
    mmd_rej = np.zeros_like(gradient_complexity,dtype='double')
    mmd_lang = np.zeros_like(gradient_complexity,dtype='double')
    
    print(gradient_complexity)
    for i, gc in enumerate(gradient_complexity):
        # Reverse Diffusion Monte Carlo
        config.p0t_method = 'ula'
        config.num_sampler_iterations = 10
        config.num_estimator_samples = gc//config.num_sampler_iterations
        samples_rdm = sample.sample(config)
        
        # Rejection
        config.p0t_method = 'rejection'
        config.num_estimator_batches = 10
        config.num_estimator_samples = gc//config.num_estimator_batches
        samples_rejection = sample.sample(config)
        
        # Langevin
        samples_langevin = samplers.ula.get_ula_samples(torch.randn_like(samples_rejection),
                                                        distribution.grad_log_prob,
                                                        .01,gc * config.disc_steps)
        
        plot_limit = 15 if config.density == 'gmm' else 3
        fig = utils.plots.plot_all_samples((samples_rejection,samples_rdm,samples_langevin),
                                           ('Ours','Reverse Diffusion Monte Carlo', 'Langevin'),
                                           plot_limit,distribution.log_prob)
        fig.savefig(f'plots/Gradient_complexity_{gc}.png', bbox_inches='tight')
        
        mmd_rdm[i] = mmd.get_mmd_squared(samples_rdm,real_samples).detach().item()
        mmd_rej[i] = mmd.get_mmd_squared(samples_rejection,real_samples).detach().item()
        mmd_lang[i] = mmd.get_mmd_squared(samples_rejection,samples_langevin).detach().item()
        
    print(mmd_lang)
    print(mmd_rej)
    print(mmd_rdm)
    # Save MMD Information    
    np.savetxt('mmd_results',(gradient_complexity, mmd_rdm,mmd_rej,mmd_lang))
    plt.plot(gradient_complexity,mmd_rdm,label='RDM')
    plt.plot(gradient_complexity,mmd_rej,label='Ours')
    plt.plot(gradient_complexity,mmd_lang,label='LMC')
    plt.title('Gradient Complexity per Discretization Step')
    plt.xlabel('Gradient Complexity')
    plt.ylabel('MMD')
    plt.legend()
    plt.savefig('mmd_results.png')
    wandb.log({'MMD Loss':plt})
    wandb.finish()


        