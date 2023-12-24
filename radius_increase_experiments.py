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
import samplers.proximal_sampler
from math import pi, sin , cos



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_gmm_radius(K,R,device):
    sigma = 1
    c = torch.ones(K,device=device)/K
    circle = torch.tensor([[cos(2*pi*i/K),sin(2*pi*i/K)] for i in range(K)]) \
        .to(dtype=torch.double, device=device) 
    offset = torch.tensor([2.,2.],dtype=torch.double, device=device)
    means = R * (circle + offset)
    variances = torch.cat([torch.eye(2).unsqueeze(0) * sigma for i in range(K)],dim=0) \
        .to(dtype=torch.double, device=device)
    return utils.densities.MixtureDistribution(c,means,variances)

def eval(config):
    setup_seed(1)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    mmd = utils.mmd.MMDLoss()

    radiuses = np.arange(2,20,step=2)
    mmd_rdm = np.zeros_like(radiuses,dtype='double')
    mmd_rej = np.zeros_like(radiuses,dtype='double')
    mmd_lang = np.zeros_like(radiuses,dtype='double')
    mmd_prox = np.zeros_like(radiuses,dtype='double')
    
    
    for i, r in enumerate(radiuses):
        distribution = get_gmm_radius(6,r,device)


        # Baseline
        tot_samples = config.num_batches * config.sampling_batch_size
        real_samples = distribution.sample(tot_samples)
    
        # Reverse Diffusion Monte Carlo
        distribution.keep_minimizer = False
        config.p0t_method = 'ula'
        config.num_estimator_samples = 100
        config.num_sampler_iterations = 100 * r//2
        config.ula_step_size = 0.1     
        config.sampling_eps = 5e-2 #RDMC is more sensitive to the early stopping
        samples_rdm = sample.sample(config,distribution)
        
        # Rejection
        distribution.keep_minimizer = True
        config.num_estimator_batches = 5
        config.num_estimator_samples = 1000
        config.p0t_method = 'rejection'
        config.sampling_eps = 5e-3
        samples_rejection = sample.sample(config,distribution)
        
        # Langevin
        distribution.keep_minimizer = False
        ula_step_size = 0.1
        num_steps_lang = 3000 # Gradient complexity for langevyn is much smaller
        samples_langevin = samplers.ula.get_ula_samples(torch.randn_like(samples_rejection),
                                                        distribution.grad_log_prob,
                                                        ula_step_size,num_steps_lang,display_pbar=False)
         
        # Proximal
        
        proximal_samples = samplers.proximal_sampler.get_samples(torch.randn_like(samples_rejection),
                                                                 distribution,
                                                                 config.proximal_M,
                                                                 config.proximal_num_iters,
                                                                 1,device
                                                                 ).squeeze(1)
        
        xlim = [2*r - 3 * r, 2*r + 3 * r]
        ylim = [2*r - 3 * r, 2*r + 3 * r]
        fig = utils.plots.plot_all_samples((real_samples, samples_rejection,samples_rdm,samples_langevin, proximal_samples),
                                        ('Ground Truth','Ours','Reverse Diffusion Monte Carlo', 'Langevin', 'Proximal'),
                                        xlim,ylim,distribution.log_prob)
        fig.savefig(f'plots/Radius_{r}.png', bbox_inches='tight')
        plt.close(fig)
        
        mmd_rdm[i] = mmd.get_mmd_squared(samples_rdm,real_samples).detach().item()
        mmd_rej[i] = mmd.get_mmd_squared(samples_rejection,real_samples).detach().item()
        mmd_lang[i] = mmd.get_mmd_squared(samples_langevin,real_samples).detach().item()
        mmd_prox[i] = mmd.get_mmd_squared(proximal_samples,real_samples).detach().item()
        
        
        
    print(mmd_lang)
    print(mmd_rej)
    print(mmd_rdm)
    # Save MMD Information
    np.savetxt('mmd_results',(radiuses, mmd_rdm,mmd_rej,mmd_lang,mmd_prox))
    fig, ax = plt.subplots()
    ax.plot(radiuses,mmd_rdm,label='RDM')
    ax.plot(radiuses,mmd_rej,label='Ours')
    ax.plot(radiuses,mmd_lang,label='LMC')
    ax.plot(radiuses,mmd_prox,label='Proximal')
    ax.set_title('MMD as a function of mode separation')
    ax.set_xlabel('radius')
    ax.set_ylabel('MMD')
    ax.legend()
    fig.savefig('plots/radius_mmd_results.png')


        