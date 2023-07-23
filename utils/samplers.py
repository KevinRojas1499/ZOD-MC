import torch
from torchdiffeq import odeint_adjoint as odeint

def get_cnf_sampler(config):
    def cnf_sampler(model):
        t0 = 0
        t1 = config.t1
        device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

        x  = torch.randn((config.batch_size,1),device=device)
        logp_diff_t1 = torch.zeros_like(x,device=device) # torch.zeros((config.batch_size,1)).to(device=device) 
        z_t, logp_diff_t = odeint(
            model,
            (x, logp_diff_t1),
            torch.tensor([t0,t1]).type(torch.float32).to(device),
            atol=config.atol,
            rtol=config.rtol,
            method='dopri5',
        )

        z_0, logp_diff_0 = z_t[-1], logp_diff_t[-1]

        return z_0, logp_diff_0

    return cnf_sampler
