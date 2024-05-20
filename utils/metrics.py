import torch
import ot
from math import log, pi
from tqdm import tqdm
from utils.densities import Distribution 
from sde_lib import SDE

class RBF_Kernel():

    def __init__(self, n_kernels=10, mul_factor=2.0):
        self.n_kernels = n_kernels
        self.bandwidth_multipliers = mul_factor ** (torch.arange(-2, -2 + n_kernels))
        
    def get_kernel_value(self, x, y):
        distances = torch.cdist(x, y) ** 2
        loss = 0
        for i in range(self.n_kernels):
            loss += torch.exp(-.5 * distances / self.bandwidth_multipliers[i]).mean()
        return loss


class MMDLoss():

    def __init__(self, kernel=RBF_Kernel()):
        self.kernel = kernel

    def get_mmd_squared(self, x, y):
        xx = self.kernel.get_kernel_value(x,x)
        xy = self.kernel.get_kernel_value(x,y)
        yy = self.kernel.get_kernel_value(y,y)
        return xx - 2 * xy + yy
    
def get_w2(samples1,samples2):
    n , m = samples1.shape[0], samples2.shape[0]
    M = ot.dist(samples1,samples2)
    a, b = torch.ones((n,),device=samples1.device) / n, torch.ones((m,),device=samples2.device) / m
    return ot.emd2(a,b,M)**.5



def hutchinson_trace_estimator(func,x, num_samples):
    with torch.enable_grad():
        x.requires_grad_(True)
        noise = torch.randn((num_samples,1,func.shape[-1]), dtype=func.dtype,device=func.device)
        eps_s = torch.sum(noise * func)
        grad = torch.autograd.grad(eps_s,x)[0]
        eps_s_eps = torch.sum(noise * grad, dim=-1)
        return torch.mean(eps_s_eps,dim=0)
    

def divergence(func,x):
    div = 0
    for i in range(x.shape[-1]):
        div += torch.autograd.grad(func[:,i].sum(),x,
                                   create_graph=True,
                                   allow_unused=True,
                                   retain_graph=True)[0][:,i:i+1]
    return div

def compute_log_normalizing_constant(dist : Distribution, sde : SDE, score_model, approx_div=False):
    num_samples = 1
    num_disc_steps = 5000
    num_hutch_samples = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = sde.T()
    time_steps = torch.linspace(0.,T-sde.delta,num_disc_steps,device=device)
    

    pbar = tqdm(range(len(time_steps) - 1),leave=False)
    xt = torch.randn((num_samples,dist.dim),device=device,requires_grad=True)
    
    logz = dist.dim/2 * log(2*pi) + torch.sum(xt**2,dim=-1,keepdim=True)/2

    for i in pbar:
        tt = time_steps[i]
        dt = time_steps[i+1] - tt
        score = score_model(xt,T-tt)
        if approx_div:
            div = hutchinson_trace_estimator(score,xt, num_hutch_samples)
        else:
            div = divergence(score,xt)
        
        logz = logz + dt * (dist.dim + div)
        xt = xt + dt * (xt + score)
    
        
    logz += dist.log_prob(xt)
    return logz