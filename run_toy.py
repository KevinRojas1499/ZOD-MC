import torch
import models.utils as mutils
import utils.optim_tools as optim_tools
from tqdm import tqdm
import losses
import utils.plots
import utils.checkpoints
import utils.samplers
import utils.densities
import os

import torch
from torch.utils.tensorboard import SummaryWriter



def train(config):
    # Create files
    log_dir = config.work_dir + "logs/"
    ckpt_path = config.work_dir + config.ckpt_dir
    samples_path = config.work_dir + config.samples_dir
    os.makedirs(ckpt_path)
    os.makedirs(samples_path)
    # Tensorboard logger
    writer = SummaryWriter(log_dir=log_dir)

    # Create Model
    model = mutils.create_model(config)
    # CNF Sampler
    cnf_sampler = utils.samplers.get_cnf_sampler(config)
    # Get optimizer
    optimizer = optim_tools.get_optimizer(config,model)
    # Target density and loss fn
    target_density = utils.densities.get_log_density_fnc(config)
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

        # Log to tensorboard
        writer.add_scalar("CNF Loss", loss_cnf, itr)
        writer.add_scalar("DSM Loss", loss_sm, itr)
        writer.add_scalar("Loss", loss, itr)



        # Save checkpoint
        if itr%config.snapshot_freq == 0:
            filename="ckpt_{}".format(itr)
            torch.save(model, ckpt_path+filename+".pt")
            z_0, _ = cnf_sampler(model,config.num_eval_samples)
            utils.plots.histogram(z_0.detach().numpy(),samples_path + filename)

        # Update bar
        t.set_description('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
        t.refresh()

    writer.flush()
    writer.close()
        


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
