import torch
import ot

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