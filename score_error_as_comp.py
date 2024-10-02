import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import click
from utils.densities import MultivariateGaussian, MixtureDistribution

import utils.score_estimators as score_estimators
from sde_lib import VP
from utils.gmm_score import get_gmm_density_at_t_no_config
from slips.samplers.mcmc import MCMCScoreEstimator
from slips.samplers.alphas import AlphaGeometric

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def to_tensor_type(x, device):
    return torch.tensor(x,device=device, dtype=torch.float32)  

def get_gmm(path, device):
    params = yaml.safe_load(open(path))
    c = to_tensor_type(params['coeffs'],device)
    means = to_tensor_type(params['means'],device)
    variances = to_tensor_type(params['variances'],device)
    return c, means, variances

def get_l2_error(real_score, generated_score):
    errors = torch.sum((real_score-generated_score)**2,dim=-1)**.5
    mean_error = torch.mean(errors)
    std = (torch.sum((errors - mean_error)**2)/(real_score.shape[0]-1))**.5
    return mean_error, std

@click.command()
@click.option('--num_samples_pt', type=int, default=1000)
@click.option('--save_folder', type=str)
@click.option('--density_params_path', type=str)
@click.option('--load_from_ckpt', is_flag=True)
def eval(num_samples_pt, save_folder, density_params_path, load_from_ckpt):
    setup_seed(1)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

    
    folder = os.path.dirname(save_folder)
    os.makedirs(folder, exist_ok=True)
    
    T = 4.
    delta = .1
    sde = VP(T,delta)
    c, means, variances = get_gmm(density_params_path, device)
    dim = means.shape[-1]
    
    dist = get_gmm_density_at_t_no_config(sde,torch.tensor([0.],device=device),c,means,variances)
    def target_log_prob_and_grad(y):
        y_ = torch.autograd.Variable(y, requires_grad=True)
        log_prob_y = dist.log_prob(y_).flatten()
        return log_prob_y, dist.grad_log_prob(y)
    alpha = AlphaGeometric(a=1.0, b=1.0)
    sigma = torch.tensor(5.0)
    
    print('slips', alpha.g(.01))
    mean_t = means * alpha.g(.01) # Doing this at the end 
    var_t = variances * alpha.g(.01)**2 + sigma**2 * torch.eye(means.shape[-1],device=variances.device)
    gaussians = [MultivariateGaussian(mean_t[i],var_t[i]) for i in range(len(c))]
    dist_slips = MixtureDistribution(c, gaussians)

    
    
    method_names = ['ZOD-MC','RDMC','RSDMC','SLIPS']
    num_complexities = 10
    num_methods = len(method_names)
    oracles = torch.arange(1,10*num_complexities+1,10, device=device) * 1000
    errors = torch.zeros((num_methods,num_complexities),device=device)
    error_std = torch.zeros((num_methods,num_complexities),device=device)
    
    
    
    T_tensor = torch.tensor([4.],device=device)
    if not load_from_ckpt:
        for i, oracle in tqdm(enumerate(oracles)):
            zodmc_score_fn = score_estimators.ZODMC_ScoreEstimator(dist,sde,device,10,oracle//10).score_estimator
            rdmc_score_normal_fn = score_estimators.RDMC_ScoreEstimator(dist,sde,device,1,oracle//100,0.1,100,True).score_estimator
            rsdmc_score_fn = score_estimators.RSDMC_ScoreEstimator(dist,sde,device,1,
                        max(1,int(np.exp(np.log(oracle.cpu())/(2 * 3)))) + 1,0.1,
                        max(1,int(np.exp(np.log(oracle.cpu())/(2 * 3)))) + 1,3,True).score_estimator
            
            print(oracle//(1000 * 50))
            slips_score = MCMCScoreEstimator(
                step_size=1e-5,
                n_mcmc_samples=1000,
                log_prob_and_grad=target_log_prob_and_grad,
                n_mcmc_chains=max(oracle//2000,1), # SLIPS has a 2 built in due to MALA
                keep_mcmc_length=int(0.5 * 100),
                use_last_mcmc_iterate=True
            )
            slips_score_fn = lambda x,t : slips_score(x,t/T_tensor,sigma,alpha) # We divide by T to kep in the desired range
            
            score_fns = [zodmc_score_fn, rdmc_score_normal_fn, rsdmc_score_fn, slips_score_fn]
            
            dist_t = get_gmm_density_at_t_no_config(sde,T_tensor,c,means,variances)
            # Baseline
            samples_t = dist_t.sample(num_samples_pt)
            true_score = dist_t.grad_log_prob(samples_t)
            for k, score in enumerate(score_fns):
                if method_names[k] == 'SLIPS':
                    true_score = dist_slips.grad_log_prob(samples_t)
                mean, std = get_l2_error(true_score, score(samples_t,T_tensor))
                errors[k,i] = mean
                error_std[k,i] = std
    else:
        errors = torch.load(os.path.join(folder,'errors.pt'))#.cpu().numpy()
        error_std = torch.load(os.path.join(folder,'std.pt'))
        
        method_names = np.load(os.path.join(folder,'method_names.npy'))
    
    # Save method names and samples
    torch.save(errors,os.path.join(folder,'errors.pt'))
    torch.save(error_std,os.path.join(folder,'std.pt'))
    
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    plt.rcParams.update({
        'font.size': 14,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    fig, ax1 = plt.subplots(1,1, figsize=(6,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']
    ax1.set_ylim(-1,4)
    for i,method in enumerate(method_names):
        if method != 'Gaussian':
            ax1.fill_between(oracles.cpu(),(errors[i] -  error_std[i]).cpu().numpy(), (errors[i] + error_std[i]).cpu().numpy(),alpha=.5)
        ax1.plot(oracles.cpu().numpy(),errors[i].cpu().numpy(),label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    ax1.set_xlabel('Oracle Complexity')
    ax1.axhline(y=0,linestyle='dotted',color='black')
    
    # ax1.set_ylim(10**0,10**7)
    ax1.set_ylabel(r'$\mathbb{E}_{p_t}[\| s(x,t) - \nabla \log p(x,t)\|]$')
    
    ax1.legend(loc='upper right')
    fig.savefig(os.path.join(folder,f'error_mmd_results_{dim}.pdf'),bbox_inches='tight')


if __name__ == '__main__':
    eval()