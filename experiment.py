import sample
import utils.densities
import matplotlib.pyplot as plt
import torch


n=3
dists = [utils.densities.RingDistribution(i+1,.01,2) for i in range(n)]
c = torch.tensor([1/n for i in range(n)])

dist = utils.densities.MixtureDistribution(c,dists)
samples = sample.zodmc(dist,1000,1,25,10000,1)
samples = samples.cpu().numpy()
plt.scatter(samples[:,0],samples[:,1])
plt.show()