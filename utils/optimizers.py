import torch
from torchmin import minimize
import time

def nesterovs_minimizer(x,gradient, eta, M,max_iters = 1500):
    d = x.shape[-1]
    A = 0
    y = x
    tau = 1
    k = 0
    mu = 1/eta - M
    L = 1/eta + M
    while torch.max(torch.sum(gradient(x)**2,dim=-1)) > (M*d)**2 and k < max_iters:
        a = (tau + (tau**2 + 4 * tau * L * A)**.5)/(2*L)
        Anext = A + a
        tx = A/Anext * y + a/Anext * x
        tauNext = tau + a * mu
        grad_tx = gradient(tx)
        yNext = tx - grad_tx/(mu + L)
        xNext = (tau * x + a * mu * tx - a * grad_tx)/tauNext
        k+=1
        A = Anext
        x = xNext
        y = yNext
        tau = tauNext
    return y, k

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