import torch
import matplotlib.pyplot as plt
import densities
from tqdm import tqdm
def sum_last_dim(x):
    return torch.sum(x,dim=-1).unsqueeze(-1)

def get_approx_minimizer(x0,gradient, threshold, al=1e-4):
    xold = x0
    xnew = x0
    k = 0
    while torch.max(gradient(xnew)) > threshold and k <3000:
        k+=1
        bek = (k-1)/(k+2)
        pk = bek * (xnew-xold)
        xold = xnew
        xnew = xnew + pk - al * gradient(xnew+pk)
    return xnew


def get_rgo_sampling(xk, yk, eta, potential, gradient, M, device):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples = xk.shape[0]
    d = xk.shape[-1]
    rej_samp = xk
    acc_samp = []
    al = 1 # We are assuming this
    delta = 1
    accepted_samples = torch.ones_like(xk)
    num_acc_samples = 0
    grad_f_eta = lambda x : gradient(x) + (x - yk)/eta
    w = get_approx_minimizer(rej_samp, grad_f_eta, (M*d)**.5)
    var = 1/(1/eta - M)
    gradw = gradient(w)
    u = (yk/eta - gradw - M * w) * var
    num_iters = 0
    while num_acc_samples < num_samples * d:
        num_iters+=1
        z = torch.randn_like(rej_samp)
        xk = u + var **.5 * accepted_samples * z 
        
        exp_h1 = potential(w) \
            + sum_last_dim(gradw * (xk-w)) \
            - M * sum_last_dim((xk-w)**2)/2 \
            - (1-al)*delta/2
        f_eta = potential(xk) # The other term gets cancelled so I removed them
        threshold1 = torch.exp(-f_eta)
        threshold2 = torch.exp(-exp_h1)
        
        rand_prob = torch.rand((num_samples,1),device=device)
        acc_idx = (accepted_samples * threshold2 * rand_prob <= threshold1)
        num_acc_samples += torch.sum(acc_idx)
        accepted_samples = (~acc_idx).long()
        
    return xk, num_iters

def get_samples(x0, eta, potential, gradient, M, num_iters, num_samples, device):
    n = x0.shape[0]
    xk = x0.repeat((num_samples,1))
    average_rejection_iters = 0
    for _ in tqdm(range(num_iters)):
        z = torch.randn_like(xk,device=device)
        yk = xk + z * eta **.5
        xk, num_iters = get_rgo_sampling(xk, yk, eta, potential, gradient, M, device)
        plt.scatter(xk[:,0].cpu(),xk[:,1].cpu())
        plt.savefig(f'./trajectory2/{_}.png')
        plt.close()
        average_rejection_iters += num_iters    
    return xk.reshape((n,num_samples,-1)), average_rejection_iters/num_iters
    

device = 'cuda'
nsamples = 1000
x = torch.randn((nsamples,2), device=device)

mean = 10
sig = 1

log_dens, grad = densities.gmm_logdensity_fnc([.25,.25,.25, .25],
                                              [[5,5],[-5,-5],[5,-5], [-5,5]],
                                              [[[1,0],[0,1]],[[1,0],[0,1]], [[1,0],[0,1]],[[1,0],[0,1]]],
                                              2,device)
def f(x):
    return -log_dens(x)
def gradf(x):
    return - grad(x)/torch.exp(log_dens(x))

num_iterations = 50
M = 1/sig**2
eta = 1/(M*2)


# times = torch.linspace(0.1,0.1,steps=1,device=device)
# rejections = torch.zeros_like(times,device=device)

# for i, t in enumerate(times):
#     print(i)
#     var_like = torch.exp(2 * t) - 1
#     potential_t = lambda x : f(x) - sum_last_dim(x**2)/(2*var_like)
#     grad_t = lambda x : gradf(x) - x/var_like
#     M_t = M + 1/var_like
#     eta = 1/(2*M_t)
#     pts, avg_rejections = get_samples(x,eta,potential_t, grad_t, M_t, num_iterations, device)
    
#     rejections[i] = avg_rejections
#     print(f"Average number of rejections {avg_rejections}")
#     plt.scatter(pts[:,0].cpu(),pts[:,1].cpu())
#     plt.savefig(f'./trajectory/{i}_{t : .3f}.png')
#     plt.close()

# plt.plot(times.cpu().numpy(), rejections.cpu().numpy())
# plt.show()