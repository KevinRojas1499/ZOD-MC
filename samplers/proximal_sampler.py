import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.densities import Distribution
from utils.plots import plot_2d_dist_with_contour
from utils.optimizers import nesterovs_minimizer

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(xk, yk, eta, dist : Distribution, M, device, initial_cond_for_minimization=None):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples, d = xk.shape #xk is assumed to be [n,d]
    al, delta = 1 ,1 # We are assuming this
    accepted_samples = torch.ones_like(xk)
    num_acc_samples = 0
    f_eta_pot = lambda x : -dist.log_prob(x) + sum_last_dim((x - yk)**2)/(2 * eta)
    grad_f_eta = lambda x : -dist.grad_log_prob(x) + (x - yk)/eta
    in_cond = xk if initial_cond_for_minimization == None else initial_cond_for_minimization
    w , k = nesterovs_minimizer(in_cond, grad_f_eta, eta, M)
    min_val = dist.log_prob(w)
    var = 1/(1/eta - M)
    gradw = -dist.grad_log_prob(w)
    u = (yk/eta - gradw - M * w) * var
    num_rejection_iters = 0
    while num_acc_samples < num_samples * d and num_rejection_iters < 20:
        num_rejection_iters+=1
        proposal = u + var **.5 * accepted_samples * torch.randn_like(xk)
        
        exp_h1 = - min_val \
            + sum_last_dim(gradw * (proposal-w)) \
            - M * sum_last_dim((proposal-w)**2)/2 \
            - (1-al)*delta/2
        f_eta = -dist.log_prob(proposal) # The other term gets cancelled so I removed them
        rand_prob = torch.rand((num_samples,1),device=device)
        acc_idx = (accepted_samples * torch.exp(-exp_h1) * rand_prob <= torch.exp(-f_eta))
        num_acc_samples = torch.sum(acc_idx)
        accepted_samples = (~acc_idx).long()
        u[acc_idx] = proposal[acc_idx]
    xk[acc_idx] = proposal[acc_idx]
    # print(num_acc_samples)
    return xk, 1 + num_rejection_iters + k, w

def get_samples(x0,dist : Distribution, M, num_iters, num_samples, device, max_grad_complexity=None):
    # x0 is [n,d] is the initialization given by the user
    # num_samples = m is the number of samples per initial condition
    # i.e. the total number of samples is n * num_samples
    # returns [n,m,d] 
    n, d = x0.shape[0], x0.shape[-1]
    xk = x0.repeat_interleave(num_samples,dim=0)
    average_rejection_iters = 0
    w = None
    eta = 1/(M*d)
    tot_grad_complexity = 0
    for _ in tqdm(range(num_iters),leave=False):
        z = torch.randn_like(xk,device=device)
        yk = xk + z * eta **.5
        xk, grad_complexity, w = get_rgo_sampling(xk, yk, eta, dist, M, device, w)
        average_rejection_iters += num_iters  
        tot_grad_complexity += grad_complexity
        if max_grad_complexity is not None and tot_grad_complexity > max_grad_complexity:
            break
        
    return xk.reshape((n,num_samples,-1))
    