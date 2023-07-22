import abc
import torch
from utils import *
import os 

class SDE():

  def __init__(self,sigma):
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

  def generate_samples_reverse(self, score_function, dimension, nsamples: int, T = 1,num = 100) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.randn((nsamples, dimension),device=device)
    time_pts = torch.linspace(T, 0, num).to(device)
    
    time_pts = self.get_edm_discretization(num, device)

    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        score = score_function(x_t,t)
        # print(x_t[0],score[0])
        tot_drift = self.f(x_t) - self.g(t)**2 * score
        tot_diffusion = self.g(t)
        # euler-maruyama step
        x_t += tot_drift * dt + tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5

    return x_t

  def get_edm_discretization(self, num, device):
      rho=7
      sigma_min = 0.002
      step_indices = torch.arange(num, dtype=torch.float64, device=device)
      t_steps = (self.sigma ** (1 / rho) + step_indices / (num - 1) * (sigma_min ** (1 / rho) - self.sigma ** (1 / rho))) ** rho
      t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
      time_pts = t_steps
      return time_pts