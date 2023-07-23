from matplotlib import pyplot as plt
import numpy as np
 
 
def histogram(x, filename):
    # Creating histogram
    L = 20
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(x, bins = np.linspace(-L,L,num=100), density=True)
    
    # Show plot
    plt.savefig(filename)