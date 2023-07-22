import torch
from torchdiffeq import odeint_adjoint as odeint
import sde_lib

def main_loss_fn(config, p0):
    def cnf_loss_fn(model):

        t0 = 0
        t1 = 10
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

        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

        logp_x = p0(z_t0) - logp_diff_t0
        loss = -logp_x.mean(0)
        return loss , z_t0
    
    def score_matching(model, samples):
        device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
        t = torch.randn((samples.shape[0],1),device=device)
        sde = sde_lib.SDE(10)
        mean = sde.marginal_prob_mean(samples,t)
        var = sde.marginal_prob_var(t)
        std = var**.5
        rand = torch.randn_like(samples)
        perturbed_samples = mean + std * rand
        score = model(t,(perturbed_samples,0))
        # score = model(t,perturbed_samples)

        loss = torch.mean((std * score[0] + rand)/std)

        return loss
        
    def loss_fn(model):
        loss_cnf, samples = cnf_loss_fn(model)
        loss_sm = score_matching(model, samples)
        return loss_cnf + 0
    return loss_fn