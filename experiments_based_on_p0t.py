import torch
import utils.integrators
import utils.score_estimators
import utils.sde_utils
import utils.gmm_score
from utils.densities import * 
from utils.plots import *
import plotly.graph_objects as go
from tqdm import tqdm

def get_run_name(config):
    return f"{config.p0t_method}"

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name + " Experiments",
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)

def to_numpy(x):
    if x.shape[-1] == 1:
        return x.squeeze(1).to('cpu').detach().numpy()
    else:
        return x[:,0].to('cpu').detach().numpy()



def get_l2_error_at_time(config, device, sde, t):
    score_fn = utils.score_estimators.get_score_function(config,sde,device)
    p0, grad = utils.gmm_score.get_gmm_density_at_t(config, sde, t, device)

    l = 6
    pts = torch.linspace(-l,l,15,device=device)
    pts = torch.cartesian_prod(pts,pts)
    est_score, avg_rejections = score_fn(pts,t)

    real_score = grad(pts)/torch.exp(p0(pts))

    return torch.log(torch.mean((real_score-est_score)**2)**.5), avg_rejections
    
def run_experiments(config):
    def get_l2_error_with_samples(config, device, sde, t, num_samples):
        config.num_estimator_samples = num_samples
        return get_l2_error_at_time(config, device, sde, t)
    

    init_wandb(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    sde = utils.sde_utils.get_sde(config)


    num_samples = [1000,3000,5000,8000, 15000]
    number_of_plots = len(num_samples)
    names = [0] * number_of_plots
    num_plots =  1 + number_of_plots
    
    N = 10
    tt = torch.linspace(0.01,sde.T(),N, device=device)
    diff_score = torch.zeros((num_plots,N))
    avg_rejections = torch.zeros((num_plots,N))

    score_error_fig = go.Figure()
    avg_rejections_fig = go.Figure()
    for i in tqdm(range(N)):
        t = tt[i]
        for j in range(len(num_samples)):
            l2_error, rejection_steps = get_l2_error_with_samples(config, device, sde, t, num_samples=num_samples[j])
            diff_score[j,i] = l2_error
            avg_rejections[j,i] = rejection_steps
            names[j] = f'{config.score_method} N= {num_samples[j]}'
        add_trace_x_not_tensor(score_error_fig, num_samples, diff_score[:,i], tt[i], dash_mode='solid')
        add_trace_x_not_tensor(avg_rejections_fig, num_samples, avg_rejections[:,i], tt[i], dash_mode='solid')
        

        wandb.log({"Score Diff": score_error_fig, "Rejections" : avg_rejections_fig})
    wandb.finish()


def add_trace_x_not_tensor(fig, xvals, yvals, name,dash_mode):
    fig.add_trace(go.Scatter(x=xvals,
                            y=yvals.detach().to('cpu').numpy(),
                            mode='lines', line=dict(dash=dash_mode), name=f"{name}"))