from matplotlib import pyplot as plt
import numpy as np
import wandb
import torch
import plotly.graph_objects as go


def histogram(x, log_density=None):
    # Creating histogram
    L = 5
    points =  np.linspace(-L,L,num=150)
    points_torch = torch.tensor(points,device='cuda').unsqueeze(-1)
    plt.hist(x, bins = points, density=True)
    if log_density is not None:
        plt.plot(points, np.exp(log_density(points_torch).to('cpu').numpy()))
    wandb.log({'Histogram': plt})

def histogram_2(x, ground_truth=None):
    # Creating histogram
    L = 20
    n = 60
    xbins=dict(start=-L,end=L, size=2*L/n)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, xbins=xbins, histnorm='probability', name='Generated'))
    fig.add_trace(go.Histogram(x=ground_truth, xbins=xbins, histnorm='probability',name='Ground Truth'))
    fig.update_layout(barmode='overlay',
                      bargap=0.1)
    fig.update_traces(opacity=0.75)
    wandb.log({'Histogram': fig})

def plot_2d_dist(data,ground_truth=None):
    L = 15
    fig = go.Figure()
    if ground_truth is not None:
        fig.add_trace(go.Scatter(x=ground_truth[:,0],y = ground_truth[:,1],mode='markers',name='Real'))
    fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1],mode='markers'))

    wandb.log({"Samples" : fig})

def to_numpy(x):
    return x.cpu().detach().numpy()
def plot_2d_dist_with_contour(data,log_prob, ground_truth=None):
    l = 15
    nn = 100
    pts = torch.linspace(-l, l, nn)
    xx , yy = torch.meshgrid(pts,pts,indexing='xy')
    pts_grid = torch.cat((xx.unsqueeze(-1),yy.unsqueeze(-1)),dim=-1).to(device='cuda')
    dens = log_prob(pts_grid).squeeze(-1).cpu()
    pts = pts.cpu().numpy()
    fig = go.Figure()
    fig.add_trace(go.Contour(z=dens, x=pts, y=pts, connectgaps=True, colorscale='darkmint'))
    fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1],mode='markers'))
    if ground_truth is not None:
        fig.add_trace(go.Scatter(x=ground_truth[:,0],y = ground_truth[:,1],mode='markers',name='Real'))
        
    wandb.log({"Samples" : fig})

def plot_all_samples(samples_array,labels,xlim, ylim,log_prob=None, take_log=False):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(1,len(samples_array), figsize=(5 * len(samples_array),6))
    if len(samples_array) == 1:
        ax = [ax]
    for i, axis in enumerate(ax):
        samp = to_numpy(samples_array[i])
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        if log_prob is not None:
            pts_x = torch.linspace(xlim[0], xlim[1], 500)
            pts_y = torch.linspace(ylim[0], ylim[1], 500)
            
            xx , yy = torch.meshgrid(pts_x,pts_y,indexing='xy')
            pts_grid = torch.cat((xx.unsqueeze(-1),yy.unsqueeze(-1)),dim=-1).to(device='cuda')
            dens = -log_prob(pts_grid).squeeze(-1).cpu().numpy()
            if take_log:
                dens = dens - np.min(dens) + 5
                dens = np.log(dens)
            pts_x, pts_y = to_numpy(pts_x), to_numpy(pts_y)
            axis.contourf(pts_x,pts_y,dens)
        
        axis.scatter(samp[:,0],samp[:,1],s=5,color='red')
        axis.set_title(labels[i])
    return fig

   
def plot_samples(config, distribution, samples,real_samples=None):
    dim = config.dimension
    if dim == 1:
        histogram(to_numpy(samples.squeeze(-1)), log_density=distribution.log_prob)
    elif dim == 2:
        real_samples = to_numpy(real_samples) if real_samples is not None else real_samples
        plot_2d_dist_with_contour(to_numpy(samples),distribution.log_prob, real_samples)
    else:
        if real_samples is not None:
            for i in range(dim):
                histogram_2(to_numpy(samples[:,i]),ground_truth=to_numpy(real_samples[:,i]))