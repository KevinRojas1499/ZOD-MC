import torch
import sde_lib
import utils.integrators
import utils.analytical_score
from utils.densities import * 
from utils.plots import *

def run_experiments(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p0 = get_log_density_fnc(config, device)
    sde = sde_lib.SDE(config)
    time_pts = torch.linspace(1,0,100)
    score_fn = utils.analytical_score.get_score_function(config,sde, device)
    # score = lambda x : score_gaussian_convolution(x,t)
    x_t = torch.randn(10000,device=device)

    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        score = score_fn(x_t,t)
        print(t)
        histogram(x_t.unsqueeze(-1).cpu().numpy(),f"./trajectory/{i}.png")

        tot_drift = sde.f(x_t) - sde.g(t)**2 * score
        tot_diffusion = sde.g(t)
        # euler-maruyama step
        x_t += tot_drift * dt + tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5

    histogram(x_t.unsqueeze(-1).cpu().numpy(),None)

    L = 1.3
    x = torch.linspace(-L, L,1000,device=device)
    sc = score_fn(x,torch.tensor([.0001],dtype=torch.float64,device=device))

    def plotable(x):
        return x.cpu().numpy()

    plt.plot(plotable(x),plotable(sc))
    plt.plot(plotable(x),-plotable(p0(x)))
    plt.legend(['convolution','true'])
    plt.show()
    plt.close()