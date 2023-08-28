import torchquad

def get_integrator(config):

    def get_trapezoid(function, a ,b):
        tp = torchquad.Trapezoid()
        return tp.integrate(fn=function,dim=1,integration_domain=[[a,b]])
    
    def get_simpson(function, a ,b):
        simp = torchquad.Simpson()
        return simp.integrate(fn=function,dim=1,integration_domain=[[a,b]])
    
    def get_monte_carlo(function, a ,b):
        mc = torchquad.MonteCarlo()
        return mc.integrate(fn=function,N=10000,dim=1,integration_domain=[[a,b]])
    
    if config.convolution_integrator == 'trap':
        return get_trapezoid
    elif config.convolution_integrator == 'simpson':
        return get_simpson
    elif config.convolution_integrator == 'mc':
        return get_monte_carlo