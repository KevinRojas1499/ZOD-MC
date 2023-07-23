import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import models.utils as mutils
import utils.optim_tools as optim_tools
from tqdm import tqdm
import losses
from torchdiffeq import odeint_adjoint as odeint
import utils.plots
import utils.checkpoints
import utils.samplers
import os


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
    # Create files
    os.makedirs(config.ckpt_dir)
    os.makedirs(config.samples_dir)

    # Create Model
    model = mutils.create_model(config)
    # CNF Sampler
    cnf_sampler = utils.samplers.get_cnf_sampler(config)
    # Get optimizer
    optimizer = optim_tools.get_optimizer(config,model)
    # Target density and loss fn
    target_density = get_logdensity_fnc(config.coeffs, config.means, config.variances)
    loss_fn = losses.main_loss_fn(config, target_density)
    # Training loop
    loss = -1 
    t = tqdm(range(1, config.train_iters + 1), leave=True)
    for itr in range(config.train_iters + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)

        loss.backward()
        optimizer.step()

        # Save checkpoint
        if itr%config.snapshot_freq == 0:
            filename="ckpt_{}".format(itr)
            torch.save(model, config.ckpt_dir+filename+".pt")
            z_0, _ = cnf_sampler(model)
            utils.plots.histogram(z_0.detach().numpy(),config.samples_dir + filename)

        # Update bar
        t.set_description('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
        t.refresh()

        


    print("--- Training has finished ---")


def eval(config):
    # Create files
    os.makedirs(config.samples_dir)

    # Load Model
    model = torch.load(config.ckpt_path)

    # CNF Sampler
    cnf_sampler = utils.samplers.get_cnf_sampler(config)

    filename="generated_samples"
    z_0, _ = cnf_sampler(model)
    utils.plots.histogram(z_0.detach().numpy(),config.samples_dir + filename)
