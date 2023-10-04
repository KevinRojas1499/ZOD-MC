import abc
import torch
import numpy as np
import utils.betas as betas

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    """Construct an SDE."""
    super().__init__()

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
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

class VP():

  def __init__(self,config):
    super().__init__()
    self.beta, self.betaprime = betas.get_beta_function(config)

  def drift(self, x,t):
    return -self.betaprime(t)* x
  
  def diffusion(self, x,t):
    return (2*self.betaprime(t))**2
    
class VE(SDE):
  def __init__(self,config):
    super().__init__()
    self.beta, self.betaprime = betas.get_beta_function(config)
    self.sigma_min = config.sigma_min
    self.sigma_max = config.sigma_max

  def drift(x,t):
    return torch.zeros_like(x)
  
  def diffusion(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    return sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
