import torch
from torchdiffeq import odeint_adjoint as odeint

def get_cnf_sampler(config):
    def cnf_sampler(model,num_samples=None):
        t0 = 0
        t1 = config.t1
        device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
        nsamples = config.batch_size
        if num_samples != None:
            nsamples = num_samples
        x  = torch.randn((nsamples,1),device=device)
        logp_diff_t1 = torch.zeros((nsamples,1)).to(device=device) 
        z_t, logp_diff_t = odeint(
            model,
            (x, logp_diff_t1),
            torch.tensor([t1,t0]).type(torch.float32).to(device),
            atol=config.atol,
            rtol=config.rtol,
            method='dopri5',
        )

        z_0, logp_diff_0 = z_t[-1], logp_diff_t[-1]

        return z_0, logp_diff_0

    return cnf_sampler
