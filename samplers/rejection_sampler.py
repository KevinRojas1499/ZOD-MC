import sys
sys.path.append('../')

import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils.optimizers import nesterovs_minimizer, gradient_descent
# import utils.densities as densities

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(xk, eta, potential, max_iters, device, threshold, minimizer=None):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples, d = xk.shape #xk is assumed to be [n,d]
    accepted_samples = torch.ones_like(xk) # 1 if rejected 0 if accepted
    num_acc_samples = 0
    w = nesterovs_minimizer(xk, potential, threshold) if \
        minimizer == None else minimizer
    num_rejection_iters = 0
    f_eta = potential(w)

    while num_acc_samples < num_samples * d and num_rejection_iters < max_iters:
        num_rejection_iters+=1
        proposal = xk + eta **.5 * accepted_samples * torch.randn_like(xk)
        
        exp_h1 = potential(proposal)
        rand_prob = torch.rand((num_samples,1),device=device)
        acc_idx = (accepted_samples * torch.exp(-f_eta) * rand_prob <= torch.exp(-exp_h1))
        num_acc_samples = torch.sum(acc_idx)
        accepted_samples = (~acc_idx).long()
        xk[acc_idx] = proposal[acc_idx]
    return xk, acc_idx, num_rejection_iters

def get_samples(y, eta, potential, num_samples, max_iters, device,threshold=1e-3,minimizer=None):
    # Sampling from potential \prop exp( - f(x) - |x-y|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = y.shape[0], y.shape[-1]
    yk = y.repeat_interleave(num_samples,dim=0)
    samples, accepted_idx, rejections = get_rgo_sampling(yk,eta,potential,max_iters,device, threshold, minimizer=minimizer)
    samples = samples.reshape((n, -1, d))
    accepted_idx = accepted_idx.reshape((n,-1,d))
    return samples, accepted_idx, rejections

# device = 'cuda'
# nsamples = 1
# x = torch.ones((nsamples,2), device=device) * 5
# mean = 10
# sig = 1
# log_dens, grad = densities.gmm_logdensity_fnc([.5,.5],
#                                               [[3,3],[-3,-3]],
#                                               [[[1,0],[0,1]],[[1,0],[0,1]]],
#                                               device)
# def f(x):
#     return -log_dens(x)
# def gradf(x):
#     return - grad(x)/torch.exp(log_dens(x))

# t = torch.linspace(.1,2, steps=50)
# etas = torch.exp(2 * t)-1
# num_iterations = torch.zeros_like(etas)
# start = time.time()
# for k in range(1):
#     for i,eta in enumerate(etas):
#         # print(i)
#         M = 1/(eta * 2)
#         __pts, num_iters = get_samples(x,eta,f, gradf, 1000, M, device) # Only one 
#         print(torch.sum(torch.isnan(__pts)))
#         pts = __pts[0].cpu().numpy()
#         print(pts.shape)
#         plt.xlim([-8,8])
#         plt.ylim([-8,8])
#         plt.scatter(pts[:,0],pts[:,1])
#         plt.savefig(f'../trajectory/{i}_{t[i] : .3f}.png')
#         plt.close()
#         num_iterations[i] = num_iters
# end = time.time()
# print(end-start)
# torch.save(num_iterations,'num_iters_changed_rule.pt')
# plt.plot(t,num_iterations)
# plt.show()