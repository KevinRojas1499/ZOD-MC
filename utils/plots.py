from matplotlib import pyplot as plt
import numpy as np
import wandb
import torch

def histogram(x, filename, log_density=None):
    # Creating histogram
    L = 15
    points =  np.linspace(-L,L,num=150)
    points_torch = torch.tensor(points)
    plt.hist(x, bins = points, density=True)
    if log_density is not None:
        plt.plot(points, np.exp(log_density(points_torch).numpy()))
    wandb.log({'my_histogram': plt})
    # Show plot
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()