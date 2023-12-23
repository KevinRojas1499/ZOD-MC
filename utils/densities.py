import abc
import torch
from torch.distributions import Normal
import yaml
from math import pi, log
from bisect import bisect_left
from random import random

class Distribution(abc.ABC):
    """ Potentials abstract class """
    def __init__(self):
        super().__init__()
        # Min
        self.potential_minimizer = None
        self.potential_min = None
        self.keep_minimizer = False # Defaults to true, set to false for simple distributions
    
    @abc.abstractmethod
    def _log_prob(self, x):
        pass
    
    def log_prob(self, x):
        # This method calls log_prob and updates the minimizer
        log_dens = self._log_prob(x)
        if self.keep_minimizer:
            xp = x.view((-1,self.dim))
            log_dens_vals = log_dens.view((-1,1))
            argmin = torch.argmin(-log_dens_vals)
            minimum = -log_dens_vals[argmin] 
            
            if self.potential_min is None or minimum < self.potential_min:
                # print(f'Updating Minimizer {xp[argmin]} {minimum}')
                self.potential_min = minimum
                self.potential_minimizer = xp[argmin]  
        return log_dens
    
    def grad_log_prob(self,x):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            torch.autograd.set_detect_anomaly(True)
            pot = self.log_prob(x)
            return torch.autograd.grad(pot.sum(),x)[0].detach()
    
    def gradient(self, x):
        return torch.exp(self.log_prob(x)) * self.grad_log_prob(x)    
class ModifiedMueller(Distribution):
    def __init__(self, A, a, b, c, XX, YY):
        super().__init__()
        self.dim = 2
        self.n = 4
        self.A = A
        self.a = a
        self.b = b
        self.c = c
        self.XX = XX
        self.YY = YY
        self.x_c = -0.033923
        self.y_c = 0.465694      
        self.beta = .1
          
    def _log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.view(-1,self.dim)
        x = xx[:,0]
        y = xx[:,1]

        V_m = 0
        for i in range(self.n):
            xi = x- self.XX[i]
            yi = y-self.YY[i]
            V_m+= self.A[i] * torch.exp(self.a[i]* xi**2 \
                    + self.b[i] * xi * yi \
                    + self.c[i] * yi**2)
        V_q = 35.0136 * (x-self.x_c)**2 + 59.8399 * (y-self.y_c)**2
        
        return -self.beta * (V_q + V_m).view(new_shape)
    
    def grad_log_prob(self, xx):
            curr_shape = list(xx.shape)
            xx = xx.view(-1,self.dim)
            x = xx[:,0]
            y = xx[:,1]

            grad_x = 0
            grad_y = 0
            for i in range(self.n):
                xi = x- self.XX[i]
                yi = y-self.YY[i]
                ee = self.A[i] * torch.exp(self.a[i]* xi**2 \
                    + self.b[i] * xi * yi \
                    + self.c[i] * yi**2)
                grad_x+=  ee * (2 * self.a[i] * xi + self.b[i] * yi)
                grad_y+=  ee * (self.b[i] * xi + 2 * self.c[i] * yi)
            
            # V_q
            grad_x += 2 * 35.0136 * (x-self.x_c)
            grad_y += 2 * 59.8399 * (y-self.y_c)
            grad_x = grad_x.unsqueeze(-1)
            grad_y = grad_y.unsqueeze(-1)
            return -self.beta * torch.cat((grad_x,grad_y),dim=-1).view(curr_shape)
class MultivariateGaussian(Distribution):
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.Q = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.linalg.inv(cov)
        self.L = torch.linalg.cholesky(self.inv_cov)
        self.log_det = torch.log(torch.linalg.det(self.cov))
        self.dim = mean.shape[0]
    
    def sample(self):
        # TODO: Make this in batches
        return self.Q @ torch.randn_like(self.mean) + self.mean
    
    def _log_prob(self,x):
        new_shape = list(x.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        x = x.view((-1,self.dim))
        shift_cov = (self.L @ (x-self.mean).T).T
        log_prob = -.5 * ( self.dim * log(2 * pi) +  self.log_det + torch.sum(shift_cov**2,dim=1)) 
        log_prob = log_prob.view(new_shape)
        return log_prob

    def grad_log_prob(self, x):
        # This is the gradient of p(x)
        curr_shape = x.shape
        x = x.view((-1,self.dim))
        grad = - (self.inv_cov @ (x - self.mean).T).T
        grad = grad.view(curr_shape)
        return grad
    
class OneDimensionalGaussian(Distribution):
    # This is a wrapper for Normal
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.dist = Normal(loc=mean, scale=cov**.5)
    
    def sample(self):
        # TODO: Make this in batches
        return self.dist.sample()
    
    def _log_prob(self,x):
        return self.dist.log_prob(x)

    def gradient(self, x):
        dens = torch.exp(self.log_prob(x))
        return - dens * (x - self.mean)/self.cov

class DoubleWell(Distribution):
    def __init__(self):
        super().__init__()
    
    def _log_prob(self,x):
        return -(x**2 - 1.)**2/2
    
    def grad_log_prob(self, x):
        return - (x**2 -1 ) * 2 * x

class GaussianMixture(Distribution):
    def __init__(self,c,means,variances):
        super().__init__()
        self.n = len(c)
        self.c = c
        self.dim = means[0].shape[0]
        self.accum = [0.]
        for i in range(self.n):
            self.accum.append(self.accum[i] + self.c[i].detach().item())
        self.accum = self.accum[1:]
        if self.dim == 1:
            self.gaussians = [OneDimensionalGaussian(means[i],variances[i]) for i in range(self.n)]
        else:
            self.gaussians = [MultivariateGaussian(means[i],variances[i]) for i in range(self.n)]

    def _log_prob(self, x):
        log_probs = []
        for i in range(self.n):
            log_probs.append( log(self.c[i]) + self.gaussians[i].log_prob(x) )
        log_probs = torch.cat(log_probs,dim=-1)
        log_dens = torch.logsumexp(log_probs,dim=-1,keepdim=True)
        return log_dens
    
    def grad_log_prob(self, x):
        log_p = self.log_prob(x)
        grad = 0
        for i in range(self.n):
            log_pi = self.gaussians[i].log_prob(x)
            grad+= self.c[i] * torch.exp(log_pi) * self.gaussians[i].grad_log_prob(x)
        return grad/(torch.exp(log_p) + 1e-8)
    
    def sample(self, num_samples):
        samples = torch.zeros(num_samples,self.dim,
                              dtype=self.gaussians[0].mean.dtype,
                              device=self.gaussians[0].mean.device)
        for i in range(num_samples):
            idx = bisect_left(self.accum, random())
            samples[i] = self.gaussians[idx].sample()
        return samples    
    
class FunnelDistribution(Distribution):
    def __init__(self, sigma, dim):
        super().__init__()
        self.dim = dim
        self.sigma = sigma

    def _log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.view(-1,self.dim)
        x1 = xx[:,0]
        y = xx[:,1:]

        log_prob = -.5 * (self.dim * log(2 * pi) + log(self.sigma**2) \
            + x1**2/self.sigma**2 + (self.dim -1) * x1 \
            + torch.sum(y**2,dim=-1) * torch.exp(-x1)
            ).view(new_shape)
        return log_prob
    
def get_distribution(config, device):
    def to_tensor_type(x):
        return torch.tensor(x,device=device, dtype=torch.double)    

    params = yaml.safe_load(open(config.density_parameters_path))

    density = config.density 
    if  density == 'gmm':
        return GaussianMixture(to_tensor_type(params['coeffs']), 
                               to_tensor_type(params['means']), 
                               to_tensor_type(params['variances']))
    elif density == 'double-well':
        return DoubleWell()
    elif density == 'mueller':
        return ModifiedMueller(to_tensor_type(params['A']),
                               to_tensor_type(params['a']), 
                               to_tensor_type(params['b']), 
                               to_tensor_type(params['c']),
                               to_tensor_type(params['XX']), 
                               to_tensor_type(params['YY']))
    elif density == 'funnel':
        return FunnelDistribution(3.,10)
    else:
        print("Density not implemented yet")
        return