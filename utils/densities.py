import abc
import torch
from torch.distributions import Normal, Laplace
import yaml
from math import pi, log

class Distribution(abc.ABC):
    """ Potentials abstract class """
    def __init__(self):
        super().__init__()
        # Min
        self.potential_minimizer = None
        self.potential_min = None
        self.keep_minimizer = False # Defaults to False, set to True for rejection sampler/optimization based algs
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
    
    def _grad_log_prob(self,x):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            torch.autograd.set_detect_anomaly(True)
            pot = self.log_prob(x)
            return torch.autograd.grad(pot.sum(),x)[0].detach()
    
    def grad_log_prob(self,x):
        return self._grad_log_prob(x)
    
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
        self.translation_x = 3.5
        self.translation_y = -6.5
        self.dilatation = 1/5
        
    def transformation(self, xx):
        x = self.dilatation * (xx[:,0] - self.translation_x)
        y = self.dilatation * (xx[:,1] - self.translation_y)
        return x,y
    
    def _log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.view(-1,self.dim)
        x,y = self.transformation(xx)

        V_m = 0
        for i in range(self.n):
            xi = x- self.XX[i]
            yi = y-self.YY[i]
            V_m+= self.A[i] * torch.exp(self.a[i]* xi**2 \
                    + self.b[i] * xi * yi \
                    + self.c[i] * yi**2)
        V_q = 35.0136 * (x-self.x_c)**2 + 59.8399 * (y-self.y_c)**2
        
        return -self.beta * (V_q + V_m).view(new_shape)
    
    def _grad_log_prob(self, xx):
        curr_shape = list(xx.shape)
        xx = xx.view(-1,self.dim)
        x,y = self.transformation(xx)

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
        return -self.beta * torch.cat((grad_x,grad_y),dim=-1).view(curr_shape) * self.dilatation
       
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

class MultivariateGaussian(Distribution):
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.Q = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.linalg.inv(cov)
        self.L = torch.linalg.cholesky(self.inv_cov)
        self.log_det = torch.log(torch.linalg.det(self.cov))
        self.dist = torch.distributions.MultivariateNormal(self.mean,self.cov)
        self.dim = mean.shape[0]
    
    def sample(self):
        # TODO: Make this in batches
        return self.Q @ torch.randn_like(self.mean) + self.mean
    
    def _log_prob(self,x):
        new_shape = list(x.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        x = x.view((-1,self.dim))
        shift_cov = (self.L.T @ (x-self.mean).T).T
        log_prob = -.5 * ( self.dim * log(2 * pi) +  self.log_det + torch.sum(shift_cov**2,dim=1)) 
        log_prob = log_prob.view(new_shape)
        return log_prob

    def _grad_log_prob(self, x):
        # This is the gradient of p(x)
        curr_shape = x.shape
        x = x.view((-1,self.dim))
        grad = - (self.inv_cov @ (x - self.mean).T).T
        grad = grad.view(curr_shape)
        return grad
  
class LaplacianDistribution(Distribution):
    def __init__(self, mean, scale):
        super().__init__()
        self.mean = mean
        self.scale = scale
        self.dim = mean.shape[-1]
        self.dist = Laplace(mean,scale)
    
    def sample(self):
        return self.dist.sample()
    
    def _log_prob(self,x):
        return self.dist.log_prob(x).sum(dim=-1,keepdim=True)

class RingDistribution(Distribution):
    # Ring Distribution
    def __init__(self, radius, scale, dim=2):
        super().__init__()
        self.radius = radius
        self.scale = scale
        self.dim = dim
    
    def sample(self,device='cuda',dtype=torch.float32):
        direction = torch.randn((1,self.dim),device=device,dtype=dtype)
        direction = direction/torch.sum(direction**2,-1,keepdim=True)**.5
        radius = self.radius + self.scale * torch.randn(1,device=device,dtype=dtype)
        return direction * radius
    
    def _log_prob(self,x):
        norm = torch.sum(x**2,dim=-1,keepdim=True)**.5
        return - (norm - self.radius)**2/(2 * self.scale**2)
    
class MixtureDistribution(Distribution):
    def __init__(self,c,distributions):
        super().__init__()
        self.n = len(c)
        self.c = c
        self.cats = torch.distributions.Categorical(c)
        self.distributions = distributions
        self.accum = [0.]
        self.dim = self.distributions[0].dim
        for i in range(self.n):
            self.accum.append(self.accum[i] + self.c[i].detach().item())
        self.accum = self.accum[1:]

    def _log_prob(self, x):
        log_probs = []
        for i in range(self.n):
            log_probs.append( log(self.c[i]) + self.distributions[i].log_prob(x) )
        log_probs = torch.cat(log_probs,dim=-1)
        log_dens = torch.logsumexp(log_probs,dim=-1,keepdim=True)
        return log_dens
    
    def _grad_log_prob(self, x):
        log_p = self.log_prob(x)
        grad = 0
        for i in range(self.n):
            log_pi = self.distributions[i].log_prob(x)
            grad+= self.c[i] * torch.exp(log_pi) * self.distributions[i].grad_log_prob(x)
        return grad/(torch.exp(log_p) + 1e-8)
    
    def sample(self, num_samples):
        one_sample = self.distributions[0].sample()
        samples = torch.zeros(num_samples,self.dim,
                              dtype=one_sample.dtype,
                              device=one_sample.device)
        for i in range(num_samples):
            idx = self.cats.sample()
            samples[i] = self.distributions[idx].sample()
        return samples

class DoubleWell(Distribution):
    def __init__(self,dim, delta):
        super().__init__()
        self.dim = dim
        self.delta = delta
        
    def _log_prob(self, x):
        return - torch.sum((x**2 - self.delta)**2,dim=-1,keepdim=True)

    def _grad_log_prob(self, x):
        return -4 * (x**2 - self.delta) * x
    
class NonContinuousPotential(Distribution):
    # For now just has discontinuities per radius
    def __init__(self, dist : Distribution):
        super().__init__()
        # Radiuses at which we should experience a jump
        self.distribution = dist
        self.dim = dist.dim
        
    def _log_prob(self, x):
        discontinuity = torch.sum(x**2,dim=-1,keepdim=True)**.5
        discontinuity[discontinuity < 5] = 0
        discontinuity[discontinuity > 11] = 0
        discontinuity*=8
        # This helps prevent problems with the backward pass
        return self.distribution._log_prob(x) - discontinuity.floor().detach() 
    
    def _grad_log_prob(self, x):
        return self.distribution._grad_log_prob(x)
    
    def sample(self,num_samples):
        N = num_samples * 100 # TODO: Don't harcode this
        s = self.distribution.sample(N)
        r = torch.rand((N,1),device=s.device)
        acc_prob = torch.exp(self.log_prob(s) - self.distribution.log_prob(s))
        acc_idx = (r < acc_prob).squeeze(-1).bool()
        return s[acc_idx][:num_samples,:]
            
            
            

class DistributionFromPotential(Distribution):
    # This is a wrapper for Normal
    def __init__(self, potential, dim):
        super().__init__()
        self.potential = potential
        self.dim = dim
    
    def _log_prob(self,x):
        return -self.potential(x)

    
def get_distribution(config, device):
    def to_tensor_type(x):
        return torch.tensor(x,device=device, dtype=torch.float32)    

    params = yaml.safe_load(open(config.density_parameters_path))

    density = config.density 
    dist = None
    if  density == 'gmm':
        c = to_tensor_type(params['coeffs'])
        means = to_tensor_type(params['means'])
        variances = to_tensor_type(params['variances'])
        n = len(c)
        if config.dimension == 1:
            gaussians = [OneDimensionalGaussian(means[i],variances[i]) for i in range(n)]
        else:
            gaussians = [MultivariateGaussian(means[i],variances[i]) for i in range(n)]

        dist = MixtureDistribution(c, gaussians)
    elif density == 'lmm':
        c = to_tensor_type(params['coeffs'])
        means = to_tensor_type(params['means'])
        scales = to_tensor_type(params['variances'])
        n = len(c)
        laplacians = [LaplacianDistribution(means[i],scales[i]) for i in range(n)]
        
        dist = MixtureDistribution(c,laplacians)
    elif density == 'rmm':
        c = to_tensor_type(params['coeffs'])
        radius = to_tensor_type(params['radius'])
        scales = to_tensor_type(params['variances'])
        n = len(c)
        rings = [RingDistribution(radius[i],scales[i],config.dimension) for i in range(n)]
        dist = MixtureDistribution(c,rings)
    elif density == 'mueller':
        dist = ModifiedMueller(to_tensor_type(params['A']),
                               to_tensor_type(params['a']), 
                               to_tensor_type(params['b']), 
                               to_tensor_type(params['c']),
                               to_tensor_type(params['XX']), 
                               to_tensor_type(params['YY']))
    elif density == 'double-well':
        dist = DoubleWell(config.dimension,3.)
    else:
        print("Density not implemented yet")
        return
    
    if config.discontinuity:
        dist = NonContinuousPotential(dist)
    
    return dist