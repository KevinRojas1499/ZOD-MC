import sys
sys.path.append('../')

import torch
import matplotlib.pyplot as plt
import time

from utils.optimizers import nesterovs_minimizer 
import utils.densities as densities

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(xk, yk, eta, potential, gradient, M, device):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples, d = xk.shape #xk is assumed to be [n,d]
    al, delta = 1 ,1 # We are assuming this
    accepted_samples = torch.ones_like(xk)
    num_acc_samples = 0
    f_eta_pot = lambda x : potential(x) + sum_last_dim((x - yk)**2)/(2 * eta)
    grad_f_eta = lambda x : gradient(x) + (x - yk)/eta
    w = nesterovs_minimizer(xk, grad_f_eta, (M*d)**.5)
    var = 1/(1/eta - M)
    gradw = gradient(w)
    u = (yk/eta - gradw - M * w) * var
    
    num_rejection_iters = 0
    while num_acc_samples < num_samples * d and num_rejection_iters < 50:
        num_rejection_iters+=1
        xk = u + var **.5 * accepted_samples * torch.randn_like(xk)
        
        exp_h1 = potential(w) \
            + sum_last_dim(gradw * (xk-w)) \
            - M * sum_last_dim((xk-w)**2)/2 \
            + sum_last_dim((xk - yk)**2)/(2*eta) \
            - (1-al)*delta/2
        f_eta = f_eta_pot(w) # The other term gets cancelled so I removed them
        rand_prob = torch.rand((num_samples,1),device=device)
        acc_idx = (accepted_samples * torch.exp(-exp_h1) * rand_prob <= torch.exp(-f_eta))
        num_acc_samples += torch.sum(acc_idx)
        accepted_samples = (~acc_idx).long()
    return xk, num_rejection_iters, w

def get_samples(y, eta, potential, gradient, num_samples, M, device):
    # Sampling from potential \prop exp( - f(x) - |x-y|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = y.shape[0], y.shape[-1]
    yk = y.repeat((num_samples,1))
    samples, rejections, _ = get_rgo_sampling(yk,yk,eta,potential,gradient,M,device)
    
    samples = samples.reshape((n, -1, d))
    return samples,rejections

device = 'cuda'
nsamples = 1
x = torch.ones((nsamples,2), device=device) * 5
mean = 10
sig = 1
log_dens, grad = densities.gmm_logdensity_fnc([.5,.5],
                                              [[3,3],[-3,-3]],
                                              [[[1,0],[0,1]],[[1,0],[0,1]]],
                                              device)
def f(x):
    return -log_dens(x)
def gradf(x):
    return - grad(x)/torch.exp(log_dens(x))

t = torch.linspace(.1,2, steps=50)
etas = torch.exp(2 * t)-1
num_iterations = torch.zeros_like(etas)
start = time.time()
for k in range(1):
    for i,eta in enumerate(etas):
        # print(i)
        M = 1/(eta * 2)
        __pts, num_iters = get_samples(x,eta,f, gradf, 1000, M, device) # Only one 
        print(torch.sum(torch.isnan(__pts)))
        pts = __pts[0].cpu().numpy()
        print(pts.shape)
        plt.xlim([-8,8])
        plt.ylim([-8,8])
        plt.scatter(pts[:,0],pts[:,1])
        plt.savefig(f'../trajectory/{i}_{t[i] : .3f}.png')
        plt.close()
        num_iterations[i] = num_iters
end = time.time()
print(end-start)
torch.save(num_iterations,'num_iters_changed_rule.pt')
plt.plot(t,num_iterations)
plt.show()