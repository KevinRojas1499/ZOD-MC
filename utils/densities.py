import abc
import torch
from torch.distributions import Normal, MultivariateNormal
import yaml
from math import pi, log

class Distribution(abc.ABC):
    """ Potentials abstract class """
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def log_prob(self, x):
        pass
    
    def grad_log_prob(self,x):
        with torch.enable_grad:
            pot = self.log_prob(x)
            return torch.autograd.grad(pot.sum(),x)[0].detach()
    
class ModifiedMueller(Distribution):
    # TODO : Implement the grad_log_prob
    def __init__(self):
        super().__init__()
        self.dim = 2
        
    def log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.view(-1,self.dim)
        x = xx[:,0]
        y = xx[:,1]
        x_c = -0.033923
        y_c = 0.465694
        V_m = -200 * torch.exp( - (x-1)**2 - 10 * y**2) \
            -100 * torch.exp(-x**2 - 10 * (y-0.5)**2) \
            -170 * torch.exp(-6.5 * (x + 0.5)**2 + 11 * (x + 0.5)* (y-1.5) - 6.5 * (y-1.5)**2) \
            +15 * torch.exp(0.7 * (x+1)**2 + 0.6 * (x+1)*(y-1) + 0.7 * (y-1)**2)
        V_q = 35.0136 * (x-x_c)**2 + 59.8399 * (y-y_c)**2
        
        return -0.1 * (V_q + V_m).view(new_shape)
    
class MultivariateGaussian():
    # This is a wrapper for Multivariate Normal
    def __init__(self, mean, cov):
        # super().__init__()
        self.mean = mean
        self.cov = cov
        self.inv_cov = torch.linalg.inv(cov)
        self.L = torch.linalg.cholesky(self.inv_cov)
        self.dim = mean.shape[0]
    
    def log_prob(self,x):
        new_shape = list(x.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        x = x.view((-1,self.dim))
        log_det = torch.log(torch.linalg.det(self.cov))
        shift_cov = (self.L @ (x-self.mean).T).T
        log_prob = -.5 * ( self.dim * log(2 * pi) +  log_det + torch.sum(shift_cov**2,dim=1)) 
        log_prob = log_prob.view(new_shape)
        return log_prob

    def gradient(self, x):
        # This is the gradient of p(x)
        curr_shape = x.shape
        x = x.view((-1,self.dim))
        dens = torch.exp(self.log_prob(x))
        grad = - dens * (self.inv_cov @ (x - self.mean).T).T
        grad = grad.view(curr_shape)
        return grad
    
class OneDimensionalGaussian(Distribution):

    # This is a wrapper for Normal
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.dist = Normal(loc=mean, scale=cov**.5)
    
    def log_prob(self,x):
        return self.dist.log_prob(x)

    def gradient(self, x):
        dens = torch.exp(self.log_prob(x))
        return - dens * (x - self.mean)/self.cov

class DoubleWell(Distribution):
    def __init__(self):
        super().__init__()
    
    def log_prob(self,x):
        return -(x**2 - 1.)**2/2
    
    def grad_log_prob(self, x):
        return - (x**2 -1 ) * 2 * x

class GaussianMixture(Distribution):
    # TODO : Implement the grad_log_prob

    def to_tensor_type(self, x, device):
        return torch.tensor(x,device=device, dtype=torch.float64)

    def __init__(self,c,means,variances, device):
        self.n = len(c)
        self.c = c
        means, variances = self.to_tensor_type(means, device),self.to_tensor_type(variances,device)
        dimension = means[0].shape[0]
        if dimension == 1:
            self.gaussians = [OneDimensionalGaussian(means[i],variances[i]) for i in range(self.n)]
        else:
            self.gaussians = [MultivariateGaussian(means[i],variances[i]) for i in range(self.n)]

    def log_prob(self, x):
        p = 0
        for i in range(self.n):
            p+= self.c[i] * torch.exp(self.gaussians[i].log_prob(x))
        return torch.log(p)
    
    def gradient(self, x):
        grad = 0
        for i in range(self.n):
            grad+= self.c[i] * self.gaussians[i].gradient(x)
        return grad
    
    def grad_log_prob(self, x):
        return self.gradient(x)/torch.exp(self.log_prob(x))
    


def get_distribution(config, device):
    params = yaml.safe_load(open(config.density_parameters_path))

    if config.density == 'gmm':
        return GaussianMixture(params['coeffs'], params['means'], params['variances'], device)
    elif config.density == 'double-well':
        return DoubleWell()
    elif config.density == 'mueller':
        return ModifiedMueller()
    else:
        print("Density not implemented yet")
        return