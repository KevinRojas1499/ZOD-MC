import torch
from tqdm import tqdm
import os

import losses
import models.utils as mutils
import utils.optim_tools as optim_tools
import utils.plots
import utils.checkpoints
import utils.samplers
import utils.densities
import utils.analytical_score
import sde_lib
import wandb

def get_run_name(config):
    return config.density + "_" + config.sampling_method + "_" + config.convolution_integrator

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name,
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)

def train(config):
    # Create files
    log_dir = config.work_dir + "logs/"
    ckpt_path = config.work_dir + config.ckpt_dir
    samples_path = config.work_dir + config.samples_dir
    os.makedirs(ckpt_path)
    os.makedirs(samples_path)
    
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    # Create Model
    model = mutils.create_model(config)
    # CNF Sampler
    cnf_sampler = utils.samplers.get_cnf_sampler(config)
    # Get optimizer
    optimizer = optim_tools.get_optimizer(config,model)
    # Target density and loss fn
    target_density, gradient = utils.densities.get_log_density_fnc(config, device)
    loss_fn = losses.main_loss_fn(config, target_density)
    # Training loop
    loss = -1 
    t = tqdm(range(1, config.train_iters + 1), leave=True)
    for itr in range(config.train_iters + 1):
        optimizer.zero_grad()
        loss_cnf, loss_sm = loss_fn(model)
        loss = loss_cnf + loss_sm
        loss.backward()
        optimizer.step()

        wandb.log({"CNF Loss": loss_cnf, "DSM Loss": loss_sm, "Loss" : loss})

        # Save checkpoint
        if itr%config.snapshot_freq == 0:
            filename="ckpt_{}".format(itr)
            torch.save(model, ckpt_path+filename+".pt")
            z_0, _ = cnf_sampler(model,config.num_eval_samples)
            utils.plots.histogram(z_0.detach().numpy(),samples_path + filename)

        # Update bar
        t.set_description('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
        t.refresh()

    wandb.finish()        


    print("--- Training has finished ---")


def eval(config):
    init_wandb(config)
    # Create files
    ckpt_path = config.work_dir + config.ckpt_path
    # os.makedirs(samples_dir)
    
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    # Get SDE:
    sde = sde_lib.SDE(config)

    # Load Model
    if config.score_method == 'convolution':
        model = utils.analytical_score.get_score_function(config, sde, device)
    else:
        model = torch.load(ckpt_path)
        model.to(device)
    # Get Sampler
    sampler = utils.samplers.get_sampler(config,sde)

    z_0 = sampler(model)
    if config.dimension == 1:
        print(z_0.shape)
        print(z_0.unsqueeze(-1).shape)
        utils.plots.histogram(to_numpy(z_0.squeeze(-1)), log_density= utils.densities.get_log_density_fnc(config,device=device)[0])
    elif config.dimension ==2:
        utils.plots.plot_2d_data(to_numpy(z_0))

def to_numpy(x):
    return x.cpu().detach().numpy()
