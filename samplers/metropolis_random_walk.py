import torch

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

# def metropolis_random_walk_iteration(x0, x, scaling, eta, potential, device):
#     # Sampling from exp(-f(x0) - (x0-scaling * x)^2/2eta)
#     num_samples, d = x0.shape #xk is assumed to be [n,d]

#     proposal = scaling * x0 + eta **.5 * torch.randn_like(x0)
#     prob1 = potential(proposal) + (sum_last_dim((proposal - scaling * x)**2) + sum_last_dim((x0- scaling * proposal)**2))/2/eta
#     prob2 = potential(x0) + (sum_last_dim((x0 - scaling * x)**2) + sum_last_dim((proposal- scaling * x0)**2))/2/eta
    
#     rand_prob = torch.rand((num_samples,1),device=device)
#     acc_prob = torch.cat((torch.ones_like(rand_prob), torch.exp(-prob1 + prob2)),dim=-1)
    
#     rejected_idx = (rand_prob > torch.min(acc_prob ,dim=-1,keepdim=True)[0]).repeat_interleave(d,dim=1) 
#     proposal[rejected_idx] = x0[rejected_idx]
#     return proposal, ~rejected_idx

def metropolis_random_walk_iteration(x0, h, potential, device):
    num_samples, d = x0.shape #xk is assumed to be [n,d]

    proposal = x0 + h **.5 * torch.randn_like(x0)
    prob1 = potential(proposal) 
    prob2 = potential(x0) 
    
    rand_prob = torch.rand((num_samples,1),device=device)
    acc_prob = torch.cat((torch.ones_like(rand_prob), torch.exp(-prob1 + prob2)),dim=-1)
    
    rejected_idx = (rand_prob > torch.min(acc_prob ,dim=-1,keepdim=True)[0]).repeat_interleave(d,dim=1) 
    proposal[rejected_idx] = x0[rejected_idx]
    return proposal, ~rejected_idx

def mala_iteration(x0,h,potential, grad_potential, device):
     # Sampling from exp(-f(x0) - (x0-scaling * x)^2/2eta)
    num_samples, d = x0.shape #xk is assumed to be [n,d]
    center = x0 - grad_potential(x0) * h
    
    proposal = center + (2*h)**.5 * torch.randn_like(x0)
    center_proposal = proposal - grad_potential(proposal) * h
    prob1 = potential(proposal) + sum_last_dim((proposal - center)**2)/(4*h)
    prob2 = potential(x0)  + sum_last_dim((x0 - center_proposal))**2/(4*h)
    
    rand_prob = torch.rand((num_samples,1),device=device)
    acc_prob = torch.cat((torch.ones_like(rand_prob), torch.exp(-prob1 + prob2)),dim=-1)
    
    rejected_idx = (rand_prob > torch.min(acc_prob ,dim=-1,keepdim=True)[0]).repeat_interleave(d,dim=1) 
    proposal[rejected_idx] = x0[rejected_idx]
    return proposal, ~rejected_idx   

def get_samples(x0, scaling, eta, potential, num_samples, num_iters, device):
    # Sampling from potential \prop exp( - f(x) - |x-scaling * x0|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = x0.shape[0], x0.shape[-1]
    x0 = x0.repeat_interleave(num_samples,dim=0)
    xk = x0.detach().clone()
    for i in range(num_iters):
        xk, acc_idx = metropolis_random_walk_iteration(xk, x0,scaling, eta,potential,device)
    xk = xk.reshape((n, -1, d))
    return xk