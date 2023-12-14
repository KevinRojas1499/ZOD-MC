import torch
import utils.score_estimators
import utils.gmm_score
import utils.integrators
import sde_lib
import utils.densities
from utils.plots import *
import plotly.graph_objects as go
from tqdm import tqdm

def get_run_name(config):
    if config.score_method == 'quotient-estimator':
        return f"{config.score_method} {config.num_estimator_samples}"
    if config.score_method == 'convolution':
        return f"{config.score_method} {config.sub_intervals_per_dim}"
    if config.score_method == 'p0t':
        return f'{config.p0t_method} {config.num_estimator_batches * config.num_estimator_samples} {config.sampling_method}'

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=f'{config.wandb_project_name} {config.dimension}d {config.density}',
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)

def to_numpy(x):
    return x.cpu().numpy()

def run_fourier_experiments(config):
    init_wandb(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = sde_lib.get_sde(config)
    dist = utils.densities.get_distribution(config,device)
    est_density = utils.score_estimators.get_score_function(config,dist,sde, device)

    x = torch.linspace(-15,15,200,device=device)
    tt = torch.linspace(0.01,sde.T(),10)
    for t in tt:
        log_prob , grad = utils.gmm_score.get_gmm_density_at_t(config,sde,t,device)
        est_dens, est_grad = est_density(x,t)
        real_dens, real_grad = torch.exp(log_prob(x)), grad(x)
        # est_dens, est_grad = est_dens[0], est_grad[0]

        dens_fig = go.Figure()
        dens_fig.add_trace(go.Scatter(x=to_numpy(x),y = to_numpy(torch.log(real_dens)),name='Real'))
        dens_fig.add_trace(go.Scatter(x=to_numpy(x),y = to_numpy(torch.log(est_dens)),name='Approximated'))
        
        grad_fig = go.Figure()
        grad_fig.add_trace(go.Scatter(x=to_numpy(x),y = to_numpy(torch.log(torch.abs(real_grad))),name='Real'))
        grad_fig.add_trace(go.Scatter(x=to_numpy(x),y = to_numpy(torch.log(torch.abs(est_grad))),name='Approximated'))
        
        score_fig = go.Figure()
        score_fig.add_trace(go.Scatter(x=to_numpy(x),y = to_numpy(real_grad/real_dens),name='Real'))
        score_fig.add_trace(go.Scatter(x=to_numpy(x),y = to_numpy(est_grad/(est_dens + config.eps_stable)),name='Approximated'))
        wandb.log({'Log Density': dens_fig, 'Log Abs Gradient': grad_fig, 'Score': score_fig})
    wandb.finish()