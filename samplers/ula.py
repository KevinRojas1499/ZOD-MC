import torch
import matplotlib.pyplot as plt
def get_ula_samples(yk, grad_log_prob, h, num_iters):
    plt.scatter(yk[:,0].cpu().numpy(),yk[:,1].cpu().numpy())
    for k in range(num_iters):
        plt.clf()
        plt.savefig(f'./trajectory/{k}')
        yk = yk + grad_log_prob(yk) * h + (2*h)**.5 * torch.randn_like(yk)
        plt.scatter(yk[:,0].cpu().numpy(),yk[:,1].cpu().numpy())
        if torch.sum(torch.isnan(yk)) > 0:
            while True:
                print("F")
    
    return yk