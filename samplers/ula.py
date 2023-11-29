import torch

def get_ula_samples(yk, grad_log_prob, h, num_iters):
    
    for k in range(num_iters):
        yk = yk + grad_log_prob(yk) * h + (2*h)**.5 * torch.randn_like(yk)
    
    return yk