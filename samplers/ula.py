import torch
from tqdm import tqdm

def get_ula_samples(yk, grad_log_prob, h, num_iters,display_pbar=False):
    for k in tqdm(range(num_iters),leave=False,disable=display_pbar):
        yk = yk + grad_log_prob(yk) * h + (2*h)**.5 * torch.randn_like(yk)
    
    return yk