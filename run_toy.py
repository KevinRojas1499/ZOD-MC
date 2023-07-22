import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import models.utils as mutils
import utils.optim_tools as optim_tools
from tqdm import tqdm
import losses
from torchdiffeq import odeint_adjoint as odeint
import utils.plots

def get_logdensity_fnc(c,means,variances):
    n = len(c)
    means, variances = torch.tensor(means), torch.tensor(variances)
    gaussians = torch.normal(means, variances)
    gaussians = [Normal(means[i],variances[i]) for i in range(n)]

    def log_density(x):
        p = 0
        for i in range(n):
            p+= c[i] * torch.exp(gaussians[i].log_prob(x))
        return torch.log(p)
    
    return log_density

def train(config):
    # Create Model
    model = mutils.create_model(config)
    # Get optimizer
    optimizer = optim_tools.get_optimizer(config,model)
    # Target density and loss fn
    target_density = get_logdensity_fnc(config.coeffs, config.means, config.variances)
    loss_fn = losses.main_loss_fn(config, target_density)
    # Training loop
    loss = -1 
    t = tqdm(range(1, config.train_iters + 1), leave=True)
    for itr in range(config.train_iters):
        optimizer.zero_grad()

        loss = loss_fn(model)

        loss.backward()
        optimizer.step()

        t.set_description('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
        t.refresh()

    print("--- Training has finished ---")
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

    z_t = z_t.cpu().detach().numpy()[-1]

    utils.plots.histogram(z_t)
