import torch
import sde_lib
import utils.integrators
import utils.analytical_score
from utils.densities import * 
from utils.plots import *
import plotly.graph_objects as go
from utils.densities import OneDimensionalGaussian

def get_run_name(config):
    return config.density + "_" + config.sampling_method + "_" + config.convolution_integrator

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name + " Experiments",
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)

def to_numpy(x):
    return x.squeeze(1).to('cpu').detach().numpy()


def to_tensor_type(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.tensor(x,device=device, dtype=torch.float64)

def gmm_logdensity_fnc(c,means,variances):
        n = len(c)
        means, variances = to_tensor_type(means),to_tensor_type(variances)

        gaussians = [OneDimensionalGaussian(means[i],variances[i]) for i in range(n)]

        def log_density(x):
            p = 0
            for i in range(n):
                p+= c[i] * torch.exp(gaussians[i].log_prob(x))
            return torch.log(p)
        
        def gradient(x):
            grad = 0
            for i in range(n):
                grad+= c[i] * gaussians[i].gradient(x)
            return grad

        return log_density, gradient
def run_experiments(config):
    init_wandb(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = sde_lib.SDE(config)

    t = torch.tensor([.5],device=device)

    score_est = utils.analytical_score.get_score_function(config,sde,device)
    c = [.5,.5]
    means=np.array([-5,5])
    variances=[1,1]
    p0, grad = gmm_logdensity_fnc(c, means * np.exp(-t.to('cpu').numpy()), variances)


    pts = torch.linspace(-10,10,500).unsqueeze(-1)
    est_dens, est_grad = score_est(pts,t)
    real_dens = torch.exp(p0(pts))
    real_grad = grad(pts)

    real_score = real_grad/real_dens
    est_score = est_grad/est_dens

    pts = pts.squeeze(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts, y=to_numpy(real_dens) ,mode='lines', name="Real"))
    fig.add_trace(go.Scatter(x=pts, y=to_numpy(est_dens),mode='lines', name="Estimated"))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pts, y=to_numpy(real_grad) ,mode='lines', name="Real"))
    fig2.add_trace(go.Scatter(x=pts, y=to_numpy(est_grad),mode='lines', name="Estimated"))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=pts, y=to_numpy(real_score) ,mode='lines', name="Real"))
    fig3.add_trace(go.Scatter(x=pts, y=to_numpy(est_score),mode='lines', name="Estimated"))

    wandb.log({"Density Diff": fig, "Grad Diff": fig2, "Score": fig3})
    wandb.finish()


