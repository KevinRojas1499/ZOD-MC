import torch
from torchmin import minimize
import time

def nesterovs_minimizer(x0,gradient, threshold=1e-5, al=1e-6):
    with torch.no_grad():
        xold = xnew = x0
        k = 0
        while torch.max(torch.sum(gradient(xnew)**2,dim=-1)**.5) > threshold and k <1500:
            # print(xnew[0])
            bek = (k-1)/(k+2)
            y = xnew + bek * (xnew-xold)
            xold = xnew
            xnew = y - al * gradient(y)
            k+=1
        return xnew

def gradient_descent(x0,gradient, threshold, al=1e-4):
    with torch.no_grad():
        xnew = x0
        k = 0
        while torch.max(torch.sum(gradient(xnew)**2,dim=-1)**.5) > threshold and k <3000:
            k+=1
            xnew = xnew - al * gradient(xnew)
        return xnew


def newton_conjugate_gradient(x0, potential,max_iters=50):
    torch.autograd.set_detect_anomaly(True)
    return minimize(
        potential, x0, 
        method='newton-cg', 
        options=dict(line_search='strong-wolfe'),
        max_iter=max_iters,
        disp=0 # Verbose level
    ).x