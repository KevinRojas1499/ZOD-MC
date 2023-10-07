import torch
import wandb
from tqdm import tqdm
import plotly.graph_objects as go

import utils.plots
import utils.samplers
import utils.densities
import utils.analytical_score
import utils.sde_utils
import utils.gmm_statistics

def get_run_name(config):
    return f"GEN-EVAL {config.density}"

def init_wandb(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project=config.wandb_project_name,
    name= get_run_name(config),
    # track hyperparameters and run metadata
    config=config
)

def sample_and_add_stats(name,config, device, weights_per_model, means_per_model):
    n_batch = config.num_batches
    dim = config.dimension
    sde = utils.sde_utils.get_sde(config)
    model = utils.analytical_score.get_score_function(config, sde, device)
    # Get Sampler
    sampler = utils.samplers.get_sampler(config,device, sde)

    samples = torch.zeros((n_batch,config.sampling_batch_size, dim))
    pbar = tqdm(range(n_batch))
    for i in pbar:
        pbar.set_description(f"Batch {i}/{n_batch}")
        samples[i] = sampler(model)
    samples = samples.view((-1,dim))
    w, error_means = utils.gmm_statistics.summarized_stats(samples)
    weights_per_model[name].append(w)
    means_per_model[name].append(error_means)

def run_experiments(config):
    init_wandb(config)
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

    sdes = ['vp','ve','edm']
    quot_estim = 'quotient-estimator'
    estimators = ['quotient-estimator','convolution']
    num_estimator_samples = [15000,30000,50000,80000, 100000]
    num_subintervals_per_dims = [151,201,301,501]
    weights_per_model = {}
    means_per_model = {}

    for sde_type in sdes:
        print(sde_type)
        for estimator in estimators:
            name = f"{sde_type}-{estimator}"
            weights_per_model[name] = []
            means_per_model[name] = []
            config.sde_type = sde_type
            config.score_method = estimator

            if estimator == quot_estim:
                for num_estimator in num_estimator_samples:
                    config.num_estimator_samples = num_estimator
                    sample_and_add_stats(name, config, device, weights_per_model, means_per_model)
            else:
                for num_subintervals in num_subintervals_per_dims:
                    config.sub_intervals_per_dim = num_subintervals
                    sample_and_add_stats(name, config, device, weights_per_model, means_per_model)

    rand_estimator_weights = go.Figure()
    rand_estimator_means = go.Figure()
    numerical_estimator_weights = go.Figure()
    numerical_estimator_means = go.Figure()
    for name, stats in weights_per_model.items():
        if quot_estim in name:
            rand_estimator_weights.add_trace(go.Scatter(x=num_estimator_samples,y=stats, name=name))
            rand_estimator_means.add_trace(go.Scatter(x=num_estimator_samples,y=means_per_model[name], name=name))
        else:
            numerical_estimator_weights.add_trace(go.Scatter(x=num_subintervals_per_dims,y=stats, name=name))
            numerical_estimator_means.add_trace(go.Scatter(x=num_subintervals_per_dims,y=means_per_model[name], name=name))


    wandb.log({"L^2 Error Weights For MC": rand_estimator_weights,
               "L^2 Error Weights For Integration": numerical_estimator_weights,
               "L^2 Error Means For MC": rand_estimator_means,
               "L^2 Error Means For Integration": numerical_estimator_means,
               })



def to_numpy(x):
    return x.cpu().detach().numpy()
