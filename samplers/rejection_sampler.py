import sys
sys.path.append('../')

import torch
import matplotlib.pyplot as plt
import utils.densities as densities
import time
import proximal_sampler

def get_samples(y, eta, potential, gradient, num_samples, M, device):
    # Sampling from potential \prop exp( - f(x) - |x-y|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = y.shape[0], y.shape[-1]
    yk = y.repeat((num_samples,1))
    samples, rejections, _ = proximal_sampler.get_rgo_sampling(yk,yk,eta,potential,gradient,M,device)
    
    samples = samples.reshape((n, -1, d))
    return samples,rejections

# device = 'cuda'
# nsamples = 1
# x = torch.zeros((nsamples,2), device=device)
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