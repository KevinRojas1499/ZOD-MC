import torch
from tqdm import tqdm

def get_ula_samples(xk, grad_log_prob, h, num_iters,display_pbar=True):
    yk = xk.detach().clone()
    for k in tqdm(range(num_iters),leave=False,disable=display_pbar):
        yk = yk + torch.nan_to_num(grad_log_prob(yk)) * h + (2*h)**.5 * torch.randn_like(yk)
    
    return yk