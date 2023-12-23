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
    w = nesterovs_minimizer(in_cond, grad_f_eta, (M*d)**.5)
    var = 1/(1/eta - M)
    gradw = -dist.grad_log_prob(w)
    u = (yk/eta - gradw - M * w) * var
    
    num_rejection_iters = 0
    while num_acc_samples < num_samples * d and num_rejection_iters < 20:
        num_rejection_iters+=1
        xk = u + var **.5 * accepted_samples * torch.randn_like(xk)
        
        exp_h1 = -dist.log_prob(w) \
            + sum_last_dim(gradw * (xk-w)) \
            - M * sum_last_dim((xk-w)**2)/2 \
            - (1-al)*delta/2
        f_eta = -dist.log_prob(xk) # The other term gets cancelled so I removed them
        rand_prob = torch.rand((num_samples,1),device=device)
        acc_idx = (accepted_samples * torch.exp(-exp_h1) * rand_prob <= torch.exp(-f_eta))
        num_acc_samples = torch.sum(acc_idx)
        accepted_samples = (~acc_idx).long()
        u[acc_idx] = xk[acc_idx]
    return xk, num_rejection_iters, w

def get_samples(x0,dist : Distribution, M, num_iters, num_samples, device):
    # x0 is [n,d] is the initialization given by the user
    # num_samples = m is the number of samples per initial condition
    # i.e. the total number of samples is n * num_samples
    # returns [n,m,d] 
    n, d = x0.shape[0], x0.shape[-1]
    xk = x0.repeat_interleave(num_samples,dim=0)
    average_rejection_iters = 0
    w = None
    eta = 1/(M*d)
    for _ in tqdm(range(num_iters),leave=False):
        z = torch.randn_like(xk,device=device)
        yk = xk + z * eta **.5
        xk, num_iters, w = get_rgo_sampling(xk, yk, eta, dist, M, device, w)
        average_rejection_iters += num_iters  
        plt.clf()
        pts_x = torch.linspace(-15,15, 100)
        pts_y = torch.linspace(-15,15, 100)
        
        xx , yy = torch.meshgrid(pts_x,pts_y,indexing='xy')
        pts_grid = torch.cat((xx.unsqueeze(-1),yy.unsqueeze(-1)),dim=-1).to(device='cuda')
        dens = -dist.log_prob(pts_grid).squeeze(-1).cpu().numpy()
        plt.scatter(xk[:,0].cpu().numpy(),xk[:,1].cpu().numpy())
        plt.contourf(pts_x.cpu().numpy(),pts_y.cpu().numpy(),dens)
        plt.savefig(f'plot/{_}.png')
        
    return xk.reshape((n,num_samples,-1))
    