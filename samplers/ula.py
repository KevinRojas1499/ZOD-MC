import torch

def get_ula_samples(yk, gradient, h, num_iters):
    
    for k in range(num_iters):
        yk = yk - gradient(yk) * h + (2*h)**.5 * torch.randn_like(yk)
    
    return yk