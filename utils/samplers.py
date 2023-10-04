import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm


def get_sampler(config, device, sde):

    def get_euler_maruyama(model):
        # TODO : Put this in the config 
        def get_edm_discretization(num, device):
            rho=7
            sigma_min = 0.002
            step_indices = torch.arange(num, dtype=torch.float64, device=device)
            t_steps = (1 ** (1 / rho) + step_indices / (num - 1) * (sigma_min ** (1 / rho) - 1 ** (1 / rho))) ** rho
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
            return t_steps

        x_t = torch.randn((config.num_samples,config.dimension),device=device)

        time_pts = get_edm_discretization(config.disc_steps, device)

        for i in tqdm(range(len(time_pts) - 1)):
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            score = model(x_t,t)
            tot_drift = sde.f(x_t) - sde.g(t)**2 * score
            tot_diffusion = sde.g(t)
            # euler-maruyama step
            x_t += tot_drift * dt + tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5

        return x_t

    if config.sampling_method == 'em':
        return get_euler_maruyama
