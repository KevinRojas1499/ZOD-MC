import torch
import utils.densities
from utils.optimizers import nesterovs_minimizer, gradient_descent

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(xk, eta, log_prob, device, threshold, minimizer=None):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples, d = xk.shape #xk is assumed to be [n,d]
    accepted_samples = torch.ones_like(xk) # 1 if rejected 0 if accepted
    potential = lambda x : - log_prob(x)
    w = nesterovs_minimizer(xk, potential, threshold) if \
        minimizer == None else minimizer
    f_eta = potential(w)
    
    proposal = xk + eta **.5 * accepted_samples * torch.randn_like(xk)
    
    exp_h1 = potential(proposal)
    rand_prob = torch.rand((num_samples,1),device=device)
    acc_idx = (accepted_samples * torch.exp(-f_eta) * rand_prob <= torch.exp(-exp_h1))
    accepted_samples = (~acc_idx).long()
    xk[acc_idx] = proposal[acc_idx]
    return xk, acc_idx

def get_samples(y, eta, distribution : utils.densities.Distribution, num_samples, device,threshold=1e-3):
    # Sampling from potential \prop exp( - f(x) - |x-y|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = y.shape[0], y.shape[-1]
    yk = y.repeat_interleave(num_samples,dim=0)
    samples, accepted_idx = get_rgo_sampling(yk,eta,distribution.log_prob,device, threshold, minimizer=distribution.potential_minimizer)
    samples = samples.reshape((n, -1, d))
    accepted_idx = accepted_idx.reshape((n,-1,d))
    return samples, accepted_idx