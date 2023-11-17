import torch
from torchmin import minimize
import time

def nesterovs_minimizer(x0,gradient, threshold, al=1e-4):
    with torch.no_grad():
        xold = x0
        xnew = x0
        k = 0
        while torch.max(gradient(xnew)) > threshold and k <500:
            k+=1
            bek = (k-1)/(k+2)
            pk = bek * (xnew-xold)
            xold = xnew
            xnew = xnew + pk - al * gradient(xnew+pk)
        return xnew

def newton_conjugate_gradient(x0, potential):
    return minimize(
        potential, x0, 
        method='newton-cg', 
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=2
    )