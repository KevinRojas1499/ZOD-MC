import torch
import utils.integrators
import utils.score_estimators
import utils.sde_utils
import utils.gmm_score
import utils.densities
from utils.plots import *
import plotly.graph_objects as go
from tqdm import tqdm

def get_run_name(config):
    if config.score_method == 'quotient-estimator':
        return f"{config.density} {config.sde_type} {config.num_estimator_samples}"
    elif config.score_method == 'convolution':
        return f"{config.density} {config.sde_type} {config.sub_intervals_per_dim}"

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name + " Experiments",
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)
    
def get_log_l2_error(real, est):
    return torch.log(torch.mean((real-est)**2)**.5)


def get_l2_error_at_time(config, device, sde, t):
    dist = utils.densities.get_distribution(config,device)
    log_prob, grad = utils.gmm_score.get_gmm_density_at_t(config,sde,t,device)
    score_fn = utils.score_estimators.get_score_function(config,dist,sde,device)

    pts = torch.linspace(-10,10,500,device=device).unsqueeze(-1)
    real_dens, real_grad= torch.exp(log_prob(pts)), grad(pts)
    est_dens, est_grad = score_fn(pts,t)
    print(torch.sum(torch.isnan(est_dens)),torch.sum(torch.isnan(est_grad)))
    real_score = real_grad/real_dens
    est_score = est_grad/est_dens 

    return get_log_l2_error(real_dens,est_dens), get_log_l2_error(real_grad, est_grad), get_log_l2_error(real_score, est_score)
    
def run_experiments(config):
    init_wandb(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = utils.sde_utils.get_sde(config)


    num_sub_intervals = [14,64,100,150,200,274,500,1000,2000]
    num_eval_pts = len(num_sub_intervals)
    N = 15
    tt = torch.linspace(0.01,sde.T(),N,device=device)
    diff_dens = torch.zeros(num_eval_pts)
    diff_grad = torch.zeros(num_eval_pts)
    diff_score = torch.zeros(num_eval_pts)
    summary_dens_fig = go.Figure()
    summary_grad_fig = go.Figure()
    summary_score_fig = go.Figure()

    for i in tqdm(range(N)):
        t = tt[i]
        for j in range(num_eval_pts):
            config.sub_intervals_per_dim = num_sub_intervals[j] # Think of a nice way to do this for different fields
            diff_dens[j], diff_grad[j], diff_score[j] =  get_l2_error_at_time(config, device, sde, t)
        add_trace_x_not_tensor(summary_dens_fig,num_sub_intervals,diff_dens,f'{t : .3f}')
        add_trace_x_not_tensor(summary_grad_fig,num_sub_intervals,diff_grad,f'{t : .3f}')
        add_trace_x_not_tensor(summary_score_fig,num_sub_intervals,diff_score,f'{t : .3f}')
        
    wandb.log({"Density Diff": summary_dens_fig, "Grad Diff": summary_grad_fig, "Score Diff": summary_score_fig})
    wandb.finish()

def add_trace(fig, xvals, yvals, name):
    fig.add_trace(go.Scatter(x=xvals.detach().to('cpu').numpy(),
                                               y=yvals.detach().to('cpu').numpy(),
                                               mode='lines', name=f"{name}"))
    
def add_trace_x_not_tensor(fig, xvals, yvals, name):
    fig.add_trace(go.Scatter(x=xvals,
                            y=yvals.detach().to('cpu').numpy(),
                            mode='lines', name=f"{name}"))