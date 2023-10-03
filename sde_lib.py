import abc
import torch
import numpy as np

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, config):
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

class OU():

  def __init__(self,config):
    sigma = config.sigma
    self.sigma = sigma
    self.f = lambda x : -sigma*x
    self.g = lambda t : (2*sigma)**.5


  def marginal_prob_mean(self, initial_mean, t):
    return initial_mean*torch.exp(-self.sigma*t)

  def marginal_prob_var(self, t, initial_var=None):
    var = 1-torch.exp(-2*self.sigma*t)
    if initial_var == None:
        return var
    else:
        return torch.exp(-2*self.sigma*t)*initial_var + var*torch.eye(initial_var.shape[0])

  def var_coord(self, x, i):
    return torch.var(x[:,i])
    
class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

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

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G