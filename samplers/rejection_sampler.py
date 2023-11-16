import torch
import matplotlib.pyplot as plt
# import densities
from tqdm import tqdm
import time

def sum_last_dim(x):
    return torch.sum(x,dim=-1).unsqueeze(-1)

def get_approx_minimizer(x0,gradient, al=1e-4):
    xold = x0
    xnew = x0
    iters = 1000
    for k in range(2,iters):
        bek = (k-1)/(k+2)
        pk = bek * (xnew-xold)
        xold = xnew
        xnew = xnew + pk - al * gradient(xnew+pk)
        
    return xnew

def get_accepted_rejected_samples(samples, threshold1, threshold2,device):
    n , m = samples.shape[0], samples.shape[1]
    rand_prob = torch.rand((n,m,1),device=device)
    acc_idx = (threshold2 * rand_prob <= threshold1).squeeze(-1)
    return acc_idx

def get_samples(y, eta, potential, gradient, num_samples, device):
    # y - [n,d]
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    n = y.shape[0]
    d = y.shape[-1]
    acc_samp = [[] for i in range(n)]
    w = get_approx_minimizer(torch.randn(d,device=device), gradient)
    batch_size = 1
    y_shaped = y.unsqueeze(1).repeat((1,batch_size,1))
    num_acc_samples = torch.zeros(n, device=device)
    num_iters = 0
    
    def f_eta(x):
        return potential(x) + sum_last_dim(x-y)**2/(2*eta)
    while 0 < num_samples:
        num_iters+=1
        # print(f'Rejection sampling iter {_}, accepted {num_acc_samples}')
        z = torch.randn_like(y_shaped,device=device)
        xk = y_shaped + eta **.5 * z 
        
        exp_h1 = potential(w) 
         # The other term gets cancelled so I removed them
        threshold1 = torch.exp(-f_eta(xk))
        threshold2 = torch.exp(-f_eta(w))
        
        acc_idx = get_accepted_rejected_samples(xk,threshold1,threshold2, device)
        num_acc_samples += torch.sum(acc_idx,dim=1)
        for i in range(n):
            acc_samp[i].append(xk[i,acc_idx[i]])
            
        if all(num_acc_samples >= num_samples):
            break
    new_samples = []
    for i in range(n):
        new_samples.append(torch.cat(acc_samp[i],dim=0))
        
    return new_samples, num_iters


# device = 'cuda'
# nsamples = 1
# x = torch.zeros((nsamples,2), device=device)

# mean = 10
# sig = 1

# log_dens, grad = densities.gmm_logdensity_fnc([.5,.5],
#                                               [[3,3],[-3,-3]],
#                                               [[[1,0],[0,1]],[[1,0],[0,1]]],
#                                               2,device)
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
#         __pts, num_iters = get_samples(x,eta,f, gradf, 500, device) # Only one 
#         pts = __pts[0].cpu().numpy()
#         # plt.xlim([-8,8])
#         # plt.ylim([-8,8])
#         # plt.scatter(pts[:,0],pts[:,1])
#         # plt.savefig(f'./trajectory/{i}_{t[i] : .3f}.png')
#         # plt.close()
#         num_iterations[i] = num_iters
# end = time.time()
# print(end-start)
# torch.save(num_iterations,'num_iters_changed_rule.pt')
# plt.plot(t,num_iterations)
# plt.show()