import torch
import utils.integrators
import utils.score_estimators
import utils.sde_utils
from utils.densities import * 
from utils.plots import *
import plotly.graph_objects as go
from tqdm import tqdm

def create_slider(tt, fig, num_plots):
    steps = []
    for i in range(len(fig.data)//num_plots):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": f"Time: {tt[i]}"}],  # layout attribute
        )
        for j in range(num_plots):
            step["args"][0]["visible"][num_plots*i+j] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]
    
    return sliders



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

def to_numpy(x):
    if x.shape[-1] == 1:
        return x.squeeze(1).to('cpu').detach().numpy()
    else:
        return x[:,0].to('cpu').detach().numpy()

def get_gmm_density_at_t(config, sde, t):
    params = yaml.safe_load(open(config.density_parameters_path))

    c, means, variances = params['coeffs'], np.array(params['means']), params['variances']
    
    scale = sde.scaling(t)
    mean_t = means * scale.to('cpu').numpy()
    var_t = variances 
    if config.sde_type == 've' or config.sde_type == 'edm':
        for i in range(len(var_t)):
            if config.sde_type == 've':
                var_t[i] = var_t[i] + t
            else:
                var_t[i] = var_t[i] + t**2
        
    return c,mean_t,var_t

def get_l2_error(real, est):
    return torch.log(torch.mean((real-est)**2)**.5)

def run_experiments(config):
    def plot_with_subintervals(config, device, sde, t, num_sub_intervals, plot_real=True,line_mode='solid'):
        config.sub_intervals_per_dim = num_sub_intervals
        name = f"{config.sde_type}, {config.score_method} N= {config.sub_intervals_per_dim}"
        return plot_at_time(config, device, sde, t, name,plot_real=plot_real, line_mode=line_mode)

    def plot_with_samples(config, device, sde, t, num_samples, plot_real=True, line_mode='solid'):
        config.num_estimator_samples = num_samples
        name = f"{config.sde_type}, {config.score_method}  samples {config.num_estimator_samples}"
        return plot_at_time(config, device, sde, t, name,plot_real=plot_real, line_mode=line_mode)
    
    def plot_at_time(config, device, sde, t, name, plot_real=True, line_mode='solid'):
        nonlocal fig, fig2, fig3
        score_est = utils.score_estimators.get_score_function(config,sde,device)
        c, mean_t, var_t = get_gmm_density_at_t(config, sde, t)
        p0, grad = gmm_logdensity_fnc(c, mean_t, var_t, config.dimension, device)

        pts = torch.linspace(-10,10,500,device=device).unsqueeze(-1)
        pts = torch.cat((pts,0*torch.ones((500,1),device=device)),dim=1)
        est_dens, est_grad = score_est(pts,t)
        real_dens = torch.exp(p0(pts))
        real_grad = grad(pts)

        real_score = real_grad/real_dens
        est_score = est_grad/est_dens 

        pts = pts[:,0].to('cpu').detach().numpy()
        # Reals
        if plot_real == True:
            fig.add_trace(go.Scatter(x=pts, y=to_numpy(real_dens) ,mode='lines', name="Real",visible=False, line=dict(dash=line_mode, width=3)))
            fig2.add_trace(go.Scatter(x=pts, y=to_numpy(real_grad) ,mode='lines', name="Real", visible=False, line=dict(dash=line_mode, width=3)))
            fig3.add_trace(go.Scatter(x=pts, y=to_numpy(real_score) ,mode='lines', name="Real", visible=False, line=dict(dash=line_mode, width=3)))

        fig.add_trace(go.Scatter(x=pts, y=to_numpy(est_dens),mode='lines', name=f"{name}", visible=False, line=dict(dash=line_mode)))
        fig2.add_trace(go.Scatter(x=pts, y=to_numpy(est_grad),mode='lines', name=f"{name}", visible=False, line=dict(dash=line_mode)))
        fig3.add_trace(go.Scatter(x=pts, y=to_numpy(est_score),mode='lines', name=f"{name}", visible=False, line=dict(dash=line_mode)))

        fig3.update_layout(yaxis_range=[-10,10])

        return {"name": name,
                "diff_dens": get_l2_error(real_dens,est_dens),
                "diff_grad": get_l2_error(real_grad, est_grad),
                "diff_score": get_l2_error(real_score, est_score) }

    init_wandb(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = utils.sde_utils.get_sde(config)


    num_samples = [500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000]
    num_sub_intervals = [5,15,25,35,51,65,81,101,121,151,175,201,225,251,275,301,401,501,601]
    number_of_plots = len(num_samples) if config.score_method == 'quotient-estimator' else len(num_sub_intervals)
    names = [0] * number_of_plots
    num_plots =  1 + number_of_plots
    
    N = 15
    tt = torch.linspace(0.01,sde.T(),N,device=device)
    diff_dens = torch.zeros((num_plots,N))
    diff_grad = torch.zeros((num_plots,N))
    diff_score = torch.zeros((num_plots,N))
    fig = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()


    for i in tqdm(range(N)):
        t = tt[i]
        if config.score_method == 'quotient-estimator':
            for j in range(len(num_samples)):
                plot_real = True if j == 0 else False
                info = plot_with_samples(config, device, sde, t, num_samples=num_samples[j],plot_real=plot_real)
                diff_dens[j,i] = info['diff_dens']
                diff_grad[j,i] = info['diff_grad']
                diff_score[j,i] = info['diff_score']
                names[j] = info['name']
        elif config.score_method == 'convolution':
            for j in range(len(num_sub_intervals)):
                plot_real = True if j == 0 else False
                info = plot_with_subintervals(config, device, sde, t, num_sub_intervals=num_sub_intervals[j],plot_real=plot_real, line_mode='solid')
                diff_dens[j,i] = info['diff_dens']
                diff_grad[j,i] = info['diff_grad']
                diff_score[j,i] = info['diff_score']
                names[j] = info['name']


    summary_score_fig = go.Figure()
    summary_dens_fig = go.Figure()
    summary_grad_fig = go.Figure()

    for i in range(N):
        xvals = num_sub_intervals
        if config.score_method == 'quotient-estimator':
            xvals = num_samples
        add_trace_x_not_tensor(summary_dens_fig, xvals, diff_dens[:,i], tt[i], dash_mode='solid')
        add_trace_x_not_tensor(summary_grad_fig, xvals, diff_grad[:,i], tt[i], dash_mode='solid')
        add_trace_x_not_tensor(summary_score_fig, xvals, diff_score[:,i], tt[i], dash_mode='solid')


    # for j in range(num_plots-1):
    #     dash_mode = 'solid'
    #     add_trace(summary_dens_fig, tt, diff_dens[j], names[j], dash_mode)
    #     add_trace(summary_grad_fig, tt, diff_grad[j], names[j], dash_mode)
    #     add_trace(summary_score_fig, tt, diff_score[j], names[j], dash_mode)

    
    # Create and add slider
    # sliders = create_slider(tt, fig, num_plots)
    # slider2 = create_slider(tt, summary_dens_fig, N)
    # summary_dens_fig.update_layout(sliders=slider2)
    # summary_grad_fig.update_layout(sliders=slider2)
    # summary_score_fig.update_layout(sliders=slider2)

    wandb.log({"Density Diff": summary_dens_fig, "Grad Diff": summary_grad_fig, "Score Diff": summary_score_fig})
    wandb.finish()

def add_trace(fig, xvals, yvals, name,dash_mode):
    fig.add_trace(go.Scatter(x=xvals.detach().to('cpu').numpy(),
                                               y=yvals.detach().to('cpu').numpy(),
                                               mode='lines', line=dict(dash=dash_mode), name=f"{name}"))
    
def add_trace_x_not_tensor(fig, xvals, yvals, name,dash_mode):
    fig.add_trace(go.Scatter(x=xvals,
                            y=yvals.detach().to('cpu').numpy(),
                            mode='lines', line=dict(dash=dash_mode), name=f"{name}"))