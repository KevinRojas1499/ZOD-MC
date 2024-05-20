import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

import utils.score_estimators as score_estimators
from sde_lib import VP
from utils.gmm_score import get_gmm_density_at_t_no_config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def to_tensor_type(x, device):
    return torch.tensor(x,device=device, dtype=torch.float32)  

def get_gmm(device):
    params = yaml.safe_load(open('config/density_parameters/5d_gmm.yaml'))
    c = to_tensor_type(params['coeffs'],device)
    means = to_tensor_type(params['means'],device)
    variances = to_tensor_type(params['variances'],device)
    return c, means, variances

def get_l2_error(real_score, generated_score):
    return torch.mean(torch.sum((real_score-generated_score)**2,dim=-1))**.5

def eval(num_samples_pt, save_folder, load_from_ckpt):
    setup_seed(1)    
    # Set up 
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

    
    folder = os.path.dirname(save_folder)
    os.makedirs(folder, exist_ok=True)
    
    T = 2.
    delta = .1
    sde = VP(T,delta)
    c, means, variances = get_gmm(device)
    dim = means.shape[-1]
    
    dist = get_gmm_density_at_t_no_config(sde,torch.tensor([0.],device=device),c,means,variances)
    zodmc_score_fn = score_estimators.ZODMC_ScoreEstimator(dist,sde,device,10*dim,10000).score_estimator
    rdmc_score_normal_fn = score_estimators.RDMC_ScoreEstimator(dist,sde,device,1,1000,0.1,100,True).score_estimator
    rsdmc_score_fn = score_estimators.RSDMC_ScoreEstimator(dist,sde,device,1,10,0.1,5,3,True).score_estimator
    
    score_fns = [zodmc_score_fn, rdmc_score_normal_fn, rsdmc_score_fn]
    method_names = ['ZOD-MC','RDMC','RSDMC']
    num_ts = 20
    ts = torch.linspace(delta, T, num_ts,device=device)
    errors = torch.zeros((3,num_ts),device=device)
    if not load_from_ckpt:
        for i, t in tqdm(enumerate(ts)):
            dist_t = get_gmm_density_at_t_no_config(sde,t,c,means,variances)
            # Baseline
            samples_t = dist_t.sample(num_samples_pt)
            true_score = dist_t.grad_log_prob(samples_t)
            for k, score in enumerate(score_fns):
                errors[k,i] = get_l2_error(true_score, score(samples_t,t))
    else:
        errors = torch.load(os.path.join(folder,'errors.pt'))#.cpu().numpy()
        method_names = np.load(os.path.join(folder,'method_names.npy'))
    
    # Save method names and samples
    torch.save(errors,os.path.join(folder,'errors.pt'))
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    plt.rcParams.update({
        'font.size': 14,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    fig, ax1 = plt.subplots(1,1, figsize=(6,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']
    print(errors)
    for i,method in enumerate(method_names):
        ax1.plot(ts.cpu().numpy(),errors[i].cpu().numpy(),label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    ax1.set_xlabel('Time')
    
    # ax1.set_ylim(10**0,10**7)
    ax1.set_ylabel(r'$\mathbb{E}_{p_t}^{1/2}[\| s(x,t) - \nabla \log p(x,t)\|^2]$')
    
    ax1.legend(loc='upper right')
    fig.savefig(os.path.join(folder,'error_mmd_results.pdf'),bbox_inches='tight')


if __name__ == '__main__':
    eval(1000,'plots/error/',True)