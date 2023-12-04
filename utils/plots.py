from matplotlib import pyplot as plt
import numpy as np
import wandb
import torch
import plotly.graph_objects as go

def histogram(x, log_density=None):
    # Creating histogram
    L = 15
    points =  np.linspace(-L,L,num=150)
    points_torch = torch.tensor(points,device='cuda')
    plt.hist(x, bins = points, density=True)
    if log_density is not None:
        plt.plot(points, np.exp(log_density(points_torch).to('cpu').numpy()))
    wandb.log({'Histogram': plt})

def histogram_2(x, ground_truth=None):
    # Creating histogram
    L = 15
    points =  np.linspace(-L,L,num=150)
    plt.hist(x, bins = points, density=True)
    if ground_truth is not None:
        plt.hist(ground_truth, bins = points, density=True)
    plt.legend(['Generated','Real'])
    wandb.log({'Histogram': plt})

def plot_2d_dist(data,ground_truth=None):
    L = 15
    fig = go.Figure()
    if ground_truth is not None:
        fig.add_trace(go.Scatter(x=ground_truth[:,0],y = ground_truth[:,1],mode='markers',name='Real'))
    fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1],mode='markers'))

    wandb.log({"Samples" : fig})
    
def plot_2d_dist_with_contour(data,log_prob):
    l = 2
    nn = 100
    pts = torch.linspace(-l, l, nn)
    xx , yy = torch.meshgrid(pts,pts,indexing='xy')
    pts_grid = torch.cat((xx.unsqueeze(-1),yy.unsqueeze(-1)),dim=-1).to(device='cuda')
    dens = log_prob(pts_grid).squeeze(-1).cpu()
    pts = pts.cpu().numpy()
    fig = go.Figure()
    fig.add_trace(go.Contour(z=dens, x=pts, y=pts, connectgaps=True))
    fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1],mode='markers'))

    wandb.log({"Samples" : fig})