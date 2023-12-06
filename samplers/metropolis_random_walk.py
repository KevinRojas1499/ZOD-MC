import torch

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(x0, xk, scaling, eta, potential, device):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples, d = xk.shape #xk is assumed to be [n,d]

    proposal = scaling * xk + eta **.5 * torch.randn_like(xk)
    prob1 = potential(proposal) + (sum_last_dim((proposal - scaling * x0)**2) + sum_last_dim((xk- scaling * proposal)**2))/2/eta
    prob2 = potential(xk) + (sum_last_dim((xk - scaling * x0)**2) + sum_last_dim((proposal- scaling * xk)**2))/2/eta
    
    rand_prob = torch.rand((num_samples,1),device=device)
    acc_prob = torch.cat((torch.ones_like(rand_prob), torch.exp(-prob1 + prob2)),dim=-1)
    
    rejected_idx = (rand_prob > torch.min(acc_prob ,dim=-1,keepdim=True)[0]).repeat_interleave(d,dim=1) 
    proposal[rejected_idx] = xk[rejected_idx]
    return proposal

def get_samples(x0, scaling, eta, potential, num_samples, num_iters, device):
    # Sampling from potential \prop exp( - f(x) - |x-scaling * x0|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = x0.shape[0], x0.shape[-1]
    x0 = x0.repeat_interleave(num_samples,dim=0)
    xk = x0.detach().clone()
    for i in range(num_iters):
        xk = get_rgo_sampling(x0, xk,scaling, eta,potential,device)
    xk = xk.reshape((n, -1, d))
    return xk