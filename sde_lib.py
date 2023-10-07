import abc
import torch
import numpy as np
import utils.transition_densities as transition_densities

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    """Construct an SDE."""
    super().__init__()

  @property
  @abc.abstractmethod
  def T(self):
    """Final Time"""
    pass

  @abc.abstractmethod
  def drift(self, x, t):
    """Returns the drift of the sde at x,t"""
    pass

  @abc.abstractmethod
  def diffusion(self, x, t):
    """Returns the diffusion of the sde at x,t"""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate samples from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def time_steps(self, num, device):
    """When discretizing the backwards SDE using Eulers method which steps to use"""

class VP():

  def __init__(self,config):
    super().__init__()
    self.scheduling = transition_densities.get_sigma_function(config)
    self.scaling = transition_densities.get_scaling_function(config)
    self.betad = config.multiplier
    self.betamin = config.bias

  def T(self):
    return 1

  def drift(self, x,t):
    return - (self.betad * t + self.betamin) * x /2
  
  def diffusion(self, x,t):
    return (self.betad * t + self.betamin)**.5
  
  def time_steps(self, n, device):
    return torch.linspace(1,0,n,device=device)
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, device=device)

    
class VE(SDE):
  def __init__(self,config):
    super().__init__()
    self.scheduling = transition_densities.get_sigma_function(config)
    self.scaling = transition_densities.get_scaling_function(config)
    self.sigma_min = config.sigma_min
    self.sigma_max = config.sigma_max

  def T(self):
    return self.sigma_max
  
  def drift(self, x,t):
    return torch.zeros_like(x)
  
  def diffusion(self, x, t):
    return 1.

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, device=device) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def time_steps(self, n, device):
    # TODO: Maybe put this in configs
    quot = (self.sigma_min/self.sigma_max)**2
    step_indices = torch.arange(n, dtype=torch.float64, device=device)/(n-1.)
    t_steps = self.sigma_max**2 * torch.pow(quot,step_indices)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps
  
class EDM(SDE):
  def __init__(self,config):
    super().__init__()
    self.scheduling = transition_densities.get_sigma_function(config)
    self.scaling = transition_densities.get_scaling_function(config)
    self.sigma_min = config.sigma_min
    self.sigma_max = config.sigma_max

  def T(self):
    return self.sigma_max
  
  def drift(self, x,t):
    return torch.zeros_like(x)
  
  def diffusion(self, x, t):
    return (2. * t)**.5

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, device=device) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def time_steps(self, n,device):
    # TODO: Maybe put this in configs
    rho=7
    step_indices = torch.arange(n, dtype=torch.float64, device=device)
    t_steps = (self.sigma_max ** (1 / rho) + step_indices / (n - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps
