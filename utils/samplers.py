import torch
from tqdm import tqdm


def get_sampler(config, device, sde):

    def get_euler_maruyama(model):

        x_t = torch.randn((config.num_samples,config.dimension),device=device)

        time_pts = sde.time_steps(config.disc_steps, device)

        for i in tqdm(range(len(time_pts) - 1)):
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            score = model(x_t, sde.beta(t))
            diffusion = sde.diffusion(x_t,t)
            tot_drift = sde.drift(x_t,t) - diffusion**2 * score
            # euler-maruyama step
            x_t += tot_drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5

        return x_t

    if config.sampling_method == 'em':
        return get_euler_maruyama
