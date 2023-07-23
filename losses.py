import torch
from torchdiffeq import odeint_adjoint as odeint
import sde_lib
import utils.samplers

def main_loss_fn(config, p0):

    sampler = utils.samplers.get_cnf_sampler(config)
    def cnf_loss_fn(model):
        z_t0, logp_diff_t0 = sampler(model)

        logp_x = p0(z_t0) - logp_diff_t0
        loss = -logp_x.mean(0)
        return loss , z_t0
    
    def score_matching(model, samples):
        device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
        t = torch.rand((samples.shape[0],1),device=device)
        sde = sde_lib.SDE(config.t1)
        mean = sde.marginal_prob_mean(samples,t)
        var = sde.marginal_prob_var(t)
        std = var**.5
        rand = torch.randn_like(samples)
        perturbed_samples = mean + std * rand
        score = model(t,(perturbed_samples,0))
        loss = torch.mean(((std * score[0] + rand)/std)**2)

        return loss
        
    def loss_fn(model):
        loss_cnf, samples = cnf_loss_fn(model)
        loss_sm = score_matching(model, samples)
        return (loss_cnf , loss_sm)
    return loss_fn