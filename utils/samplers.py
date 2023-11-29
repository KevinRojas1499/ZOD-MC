import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import samplers.ula as ula
def get_sampler(config, device, sde):
    torch.manual_seed(123)
    
    def plot_trajectory(x_t, i, t):
        plt.xlim([-8,8])
        plt.ylim([-8,8])
        plt.plot(x_t[:,0].cpu(), x_t[:,1].cpu(),'.')
        plt.savefig(f'./trajectory/{i}_{t : .3f}.png')
        plt.close()
    
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
            plot_trajectory(x_t, i, t)
        pbar.close()
        return x_t
    
    def get_exponential_integrator(model):

        x_t = sde.prior_sampling((config.sampling_batch_size,config.dimension),device=device)

        time_pts = sde.time_steps(config.disc_steps, device, config.sampling_method)
        T = sde.T()
        print(time_pts)
        pbar = tqdm(range(len(time_pts) - 1),leave=False)
        for i in pbar:
            t = time_pts[i]
            dt = time_pts[i] - time_pts[i+1]
            score = model(x_t, T - t)
            e_h = torch.exp(dt)
            # exponential integrator step
            x_t = e_h * x_t + 2 * (1- e_h) * score + (2*(1-e_h**2))**.5 * torch.randn_like(x_t)
            plot_trajectory(x_t, i, t)
        pbar.close()
        return x_t
    if config.sampling_method == 'em':
        return get_euler_maruyama
    if config.sampling_method == 'ei':
        return get_exponential_integrator
