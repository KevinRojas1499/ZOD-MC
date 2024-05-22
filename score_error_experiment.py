import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import click

import utils.score_estimators as score_estimators
from sde_lib import VP
from utils.gmm_score import get_gmm_density_at_t_no_config

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
    std = torch.mean((errors - mean_error)**2)**.5
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
    zodmc_score_fn = score_estimators.ZODMC_ScoreEstimator(dist,sde,device,10*dim,10000).score_estimator
    rdmc_score_normal_fn = score_estimators.RDMC_ScoreEstimator(dist,sde,device,1,1000,0.1,100,True).score_estimator
    rsdmc_score_fn = score_estimators.RSDMC_ScoreEstimator(dist,sde,device,1,10,0.1,5,3,True).score_estimator
    def standard_gaussian_score(x,t):
        return -x
    
    score_fns = [zodmc_score_fn, rdmc_score_normal_fn, rsdmc_score_fn]
    method_names = ['ZOD-MC','RDMC','RSDMC']
    num_ts = 20
    num_methods = len(method_names)
    ts = torch.linspace(delta, T, num_ts,device=device)
    errors = torch.zeros((num_methods,num_ts),device=device)
    error_std = torch.zeros((num_methods,num_ts),device=device)
    
    if not load_from_ckpt:
        for i, t in tqdm(enumerate(ts)):
            dist_t = get_gmm_density_at_t_no_config(sde,t,c,means,variances)
            # Baseline
            samples_t = dist_t.sample(num_samples_pt)
            true_score = dist_t.grad_log_prob(samples_t)
            for k, score in enumerate(score_fns):
                mean, std = get_l2_error(true_score, score(samples_t,t))
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
            ax1.fill_between(ts.cpu(),(errors[i] -  error_std[i]).cpu().numpy(), (errors[i] + error_std[i]).cpu().numpy(),alpha=.5)
        ax1.plot(ts.cpu().numpy(),errors[i].cpu().numpy(),label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    ax1.set_xlabel('Time')
    ax1.axhline(y=0,linestyle='dotted',color='black')
    
    # ax1.set_ylim(10**0,10**7)
    ax1.set_ylabel(r'$\mathbb{E}_{p_t}[\| s(x,t) - \nabla \log p(x,t)\|]$')
    
    ax1.legend(loc='upper right')
    fig.savefig(os.path.join(folder,f'error_mmd_results_{dim}.pdf'),bbox_inches='tight')


if __name__ == '__main__':
    eval()