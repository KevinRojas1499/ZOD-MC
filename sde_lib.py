import torch

class SDE():

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
