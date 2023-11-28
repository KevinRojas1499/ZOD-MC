import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import samplers.ula as ula
def get_sampler(config, device, sde):
    torch.manual_seed(123)
    def get_euler_maruyama(model):

        x_t = sde.prior_sampling((config.sampling_batch_size,config.dimension),device=device)

        time_pts = sde.time_steps(config.disc_steps, device)
        pbar = tqdm(range(len(time_pts) - 1),leave=False)
        for i in pbar:
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            score = model(x_t, t)
            diffusion = sde.diffusion(x_t,t)
            tot_drift = sde.drift(x_t,t) - diffusion**2 * score
            # euler-maruyama step
            x_t += tot_drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
            plt.plot(x_t[:,0].cpu(), x_t[:,1].cpu(),'.')
            plt.savefig(f'./trajectory/{i}_{t : .3f}.png')
            plt.close()
        pbar.close()
        return x_t
    
    if config.sampling_method == 'em':
        return get_euler_maruyama
