import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
def get_sampler(config, device, sde):
    torch.manual_seed(123)
    
    def plot_trajectory(x_t, i, t):
        lim = 3 if config.density == 'mueller' else 15
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
        plt.scatter(x_t[:,0].cpu(), x_t[:,1].cpu(),s=2)
        plt.savefig(f'./trajectory/{i}_{t : .3f}.png')
        plt.close()
    
    def get_euler_maruyama(model):
        x_t = sde.prior_sampling((config.sampling_batch_size,config.dimension),device=device)

        time_pts = sde.time_steps(config.disc_steps, device)
        # torch.set_printoptions(precision=3,sci_mode=False)
        # print(time_pts)
        pbar = tqdm(range(len(time_pts) - 1),leave=False)
        T = sde.T()
        for i in pbar:
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            score = model(x_t, T- t)
            diffusion = sde.diffusion(x_t,T - t)
            tot_drift = - sde.drift(x_t,T - t) + diffusion**2 * score
            # euler-maruyama step    print(samples.shape)

            x_t += tot_drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
            
            # plot_trajectory(x_t, i, t)
        pbar.close()
        return x_t
    
    def get_exponential_integrator(model):

        x_t = sde.prior_sampling((config.sampling_batch_size,config.dimension),device=device)

        time_pts = sde.time_steps(config.disc_steps, device)
        T = sde.T()
        # torch.set_printoptions(precision=3,sci_mode=False)
        # print(time_pts)
        pbar = tqdm(range(len(time_pts) - 1),leave=False)
        for i in range(len(time_pts) - 1):
            t = time_pts[i]
            dt = time_pts[i+1] - time_pts[i]
            score = model(x_t, T - t)
            e_h = torch.exp(dt)
            # exponential integrator step
            x_t = e_h * x_t + 2 * (e_h - 1) * score + ((e_h**2 - 1))**.5 * torch.randn_like(x_t)
            # plot_trajectory(x_t, i, t)
        pbar.close()
        return x_t
    if config.sampling_method == 'em':
        return get_euler_maruyama
    if config.sampling_method == 'ei':
        return get_exponential_integrator
