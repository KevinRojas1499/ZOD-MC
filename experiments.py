import torch
import sde_lib
import utils.integrators
import utils.analytical_score
from utils.densities import * 
from utils.plots import *
import plotly.graph_objects as go
from utils.densities import OneDimensionalGaussian, MultivariateGaussian
from tqdm import tqdm


def get_run_name(config):
    return f"{config.density} {config.score_method} {config.gradient_estimator}"

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


def to_tensor_type(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.tensor(x,device=device, dtype=torch.float64)

def gmm_logdensity_fnc(c,means,variances):
        n = len(c)
        means, variances = to_tensor_type(means),to_tensor_type(variances)

        if len(means.shape) == 1:
            gaussians = [OneDimensionalGaussian(means[i],variances[i]) for i in range(n)]
        else:
            gaussians = [MultivariateGaussian(means[i],variances[i]) for i in range(n)]

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
    def plot_at_time(config, device, sde, t):
        nonlocal fig, fig2, fig3
        score_est = utils.analytical_score.get_score_function(config,sde,device)
        params = yaml.safe_load(open(config.density_parameters_path))

        c = params['coeffs']
        means=np.array(params['means'])
        variances= params['variances']

        mean_t = means * np.exp(-4*t.to('cpu').numpy())
        p0, grad = gmm_logdensity_fnc(c, mean_t, variances)


        pts = torch.linspace(-10,10,500).unsqueeze(-1)
        pts = torch.cat((pts,0*torch.ones((500,1))),dim=1)
        est_dens, est_grad = score_est(pts,t)
        real_dens = torch.exp(p0(pts))
        real_grad = grad(pts)

        real_score = real_grad/real_dens
        est_score = est_grad/est_dens 

        pts = pts[:,0]
        fig.add_trace(go.Scatter(x=pts, y=to_numpy(real_dens) ,mode='lines', name="Real",visible=False))
        fig.add_trace(go.Scatter(x=pts, y=to_numpy(est_dens),mode='lines', name="Estimated", visible=False))

        fig2.add_trace(go.Scatter(x=pts, y=to_numpy(real_grad) ,mode='lines', name="Real", visible=False))
        fig2.add_trace(go.Scatter(x=pts, y=to_numpy(est_grad),mode='lines', name="Estimated", visible=False))

        fig3.add_trace(go.Scatter(x=pts, y=to_numpy(real_score) ,mode='lines', name="Real", visible=False))
        fig3.add_trace(go.Scatter(x=pts, y=to_numpy(est_score),mode='lines', name="Estimated", visible=False))



    init_wandb(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = sde_lib.SDE(config)

    N = 50
    tt = torch.linspace(0,1,N)
    fig = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()

    for i in tqdm(range(N)):
        t = tt[i]
        plot_at_time(config, device, sde, t)

    # Create and add slider
    steps = []
    for i in range(len(fig.data)//2):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": f"Time: {tt[i]*4}"}],  # layout attribute
        )
        step["args"][0]["visible"][2*i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][2*i+1] = True  # Toggle i'th trace to "visible"

        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig2.update_layout(sliders=sliders)
    fig3.update_layout(sliders=sliders)

    wandb.log({"Density Diff": fig, "Grad Diff": fig2, "Score": fig3})
    wandb.finish()

