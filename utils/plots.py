from matplotlib import pyplot as plt
import numpy as np
import wandb
import torch
import plotly.graph_objects as go

def histogram(x, log_density=None):
    # Creating histogram
    L = 15
    points =  np.linspace(-L,L,num=150)
    points_torch = torch.tensor(points)
    plt.hist(x, bins = points, density=True)
    if log_density is not None:
        plt.plot(points, np.exp(log_density(points_torch).numpy()))
    wandb.log({'my_histogram': plt})

def plot_2d_dist(data):
    L = 15
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1],mode='markers'))

    wandb.log({"Sampling" : fig})