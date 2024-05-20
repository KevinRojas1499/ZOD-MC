import torch
from tqdm import tqdm

from utils.densities import Distribution

def parallel_tempering(distribution : Distribution,
                                initial_cond, betas,num_iters,h, device='cuda'):
    # Betas is the array of inverse temperatures
    d = distribution.dim
    num_chains = betas.shape[0]
    betas = betas.reshape(num_chains,1,1)
    xchains = initial_cond.expand(num_chains,*initial_cond.shape).clone().to(device) if len(initial_cond.shape) == 2 else initial_cond
    for i in tqdm(range(num_iters), leave=False):
        xk = xchains
        center = xk + h * betas * distribution.grad_log_prob(xk).nan_to_num()
        proposal = center + (2*h)**.5 * torch.randn_like(xk)
        center_proposal = proposal + h * betas * distribution.grad_log_prob(proposal)
        xchains = proposal
        prob1 = distribution.log_prob(proposal) - torch.sum((proposal - center)**2,dim=-1,keepdim=True)/(4*h)
        prob2 = distribution.log_prob(xk)  - torch.sum((xk - center_proposal)**2,dim=-1,keepdim=True)/(4*h)
        acc_rate = torch.exp(prob1 - prob2)
        acc_rate = torch.min(torch.ones_like(acc_rate),acc_rate) 
        acc = torch.rand_like(acc_rate) < acc_rate
        acc = acc.expand((-1,-1,d))
        xchains[acc] = proposal[acc]
        for k in range(1,num_chains):
            xii = xchains[k-1]
            xi = xchains[k]
            acc_rate = torch.exp((betas[k] - betas[k-1]) * distribution.log_prob(xii)
                                +(betas[k-1] - betas[k]) * distribution.log_prob(xi))
            acc_rate = torch.min(torch.ones_like(acc_rate),acc_rate)
            acc = torch.rand_like(acc_rate) < acc_rate
            acc = acc.expand((-1,d))
            xchains[k-1][acc] = xi[acc]
            xchains[k][acc] = xii[acc]         
    
    return xchains[num_chains-1], xchains