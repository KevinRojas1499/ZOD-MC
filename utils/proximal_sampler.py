import torch
import matplotlib.pyplot as plt
# import utils.densities

def sum_last_dim(x):
    return torch.sum(x,dim=-1).unsqueeze(-1)

def get_approx_minimizer(x0,gradient, al=1e-4):
    xold = x0
    xnew = x0
    k = 1000
    for _ in range(2,k):
        bek = (k-1)/(k+2)
        pk = bek * (xnew-xold)
        xold = xnew
        xnew = xnew + pk - al * gradient(xnew+pk)
        
    return xnew

def get_accepted_rejected_samples(samples, threshold1, threshold2,device):
    n = samples.shape[0]
    rand_prob = torch.rand((n,1),device=device)
    acc_idx = (threshold2 * rand_prob <= threshold1).squeeze(-1)
    return samples[acc_idx], samples[~acc_idx], ~acc_idx

def get_rgo_sampling(x0, yk, eta, potential, gradient, M, device):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples = x0.shape[0]
    rej_samp = x0
    acc_samp = []
    al = 1 # We are assuming this
    delta = 1
    num_acc_samples = 0
    
    grad_f_eta = lambda x : gradient(x) + (x - yk)/eta
    w = get_approx_minimizer(rej_samp, grad_f_eta)
    var = 1/(1/eta - M)
    gradw = gradient(w)
    u = (yk/eta - gradw - M * w) * var
    
    while num_acc_samples < num_samples:
        # print(f'Rejection sampling iter {_}, accepted {num_acc_samples}')
        z = torch.randn_like(rej_samp)
        xk = u + var **.5 * z 
        
        exp_h1 = potential(w) \
            + sum_last_dim(gradw * (xk-w)) \
            - M * sum_last_dim((xk-w)**2)/2 \
            - (1-al)*delta/2
        f_eta = potential(xk) # The other term gets cancelled so I removed them
        threshold1 = torch.exp(-f_eta)
        threshold2 = torch.exp(-exp_h1)
        
        new_acc, rej, rej_idx = get_accepted_rejected_samples(xk,threshold1,threshold2, device)
        if new_acc.shape[0] > 0:
            num_acc_samples += new_acc.shape[0]
            acc_samp.append(new_acc)
            if num_acc_samples >= num_samples:
                break
        rej_samp = rej
        u = u[rej_idx]
        w = w[rej_idx]
        gradw = gradw[rej_idx]
        
    return torch.cat(acc_samp,dim=0)

def get_proximal_sampler(x0, eta, potential, gradient, M, device):
    k = 50
    xk = x0
    for _ in range(k):
        print(f"Iteration {_}")
        z = torch.randn_like(xk,device=device)
        yk = xk + z * eta **.5
        xk = get_rgo_sampling(xk, yk, eta, potential, gradient, M, device)
    return xk
    

# device = 'cuda'
# nsamples = 1000
# x = torch.randn((nsamples,2), device=device)

# mean = 10
# sig = 1

# log_dens, grad = densities.gmm_logdensity_fnc([.25,.25,.25, .25],
#                                               [[5,5],[-5,-5],[5,-5], [-5,5]],
#                                               [[[1,0],[0,1]],[[1,0],[0,1]], [[1,0],[0,1]],[[1,0],[0,1]]],
#                                               2,device)
# def f(x):
#     return -log_dens(x)
# def gradf(x):
#     return - grad(x)/torch.exp(log_dens(x))

# M = 1/sig**2
# eta = 1/(M*2)
# print(eta)
# pts = get_proximal_sampler(x,eta,f, gradf, M, device)
# print(torch.mean(pts,dim=0))
# pts = pts.cpu().numpy()
# x = (sig * x + mean).cpu().numpy()
# plt.scatter(pts[:,0],pts[:,1])
# # plt.scatter(x[:,0],x[:,1])
# # plt.legend(['app','real'])
# plt.show()