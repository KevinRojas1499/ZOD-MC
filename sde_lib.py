import abc
import torch
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

class VP(SDE):

  def __init__(self,config):
    super().__init__()
    self.betad = config.multiplier
    self.betamin = config.bias
    self.delta = config.sampling_eps

  def scheduling(self, t):
        return (torch.exp(self.betad * t**2/2 + self.betamin *t) -1)**.5
  def scaling(self, t):
    return torch.exp(-(self.betad * t**2/2 + self.betamin * t)/2)
  
  def T(self):
    return 10.

  def drift(self, x,t):
    return - (self.betad * t + self.betamin) * x /2
  
  def diffusion(self, x,t):
    return (self.betad * t + self.betamin)**.5
  
  def time_steps(self, n, device):
    from math import exp, log
    c = 1.6 * (exp(log(self.T()/self.delta)/n) - 1)
    t_steps = torch.zeros(n,device=device)
    t_steps[0] = self.delta
    exp_step = True
    for i in range(1,n):
      if exp_step:
        t_steps[i] = t_steps[i-1] + c * t_steps[i-1]
        if t_steps[i] >= 1:
          c = (self.T() - t_steps[i-1])/(n-i)
          t_steps[i] = t_steps[i-1] + c
          exp_step = False
      else:
        t_steps[i] = t_steps[i-1] + c
    
    t_steps[-1] = self.T()
    t_steps = self.T() - t_steps  
    t_steps = torch.flip(t_steps,dims=(0,))
    return t_steps
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, device=device)

def get_sde(config):
    if config.sde_type == 'vp':
        return VP(config)