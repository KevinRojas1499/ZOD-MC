import torchquad

def get_integrator(config):
    dimension = config.dimension
    def get_trapezoid(function, a ,b):
        tp = torchquad.Trapezoid()
        return tp.integrate(fn=function,dim=dimension,integration_domain=[[a,b]]  * dimension)
    
    def get_simpson(function, a ,b):
        simp = torchquad.Simpson()
        return simp.integrate(fn=function,N=100,dim=dimension,integration_domain=[[a,b]]  * dimension)
    
    def get_monte_carlo(function, a ,b):
        mc = torchquad.MonteCarlo()
        return mc.integrate(fn=function,N=1000,dim=dimension,integration_domain=[[a,b]] * dimension)
    
    if config.convolution_integrator == 'trap':
        return get_trapezoid
    elif config.convolution_integrator == 'simpson':
        return get_simpson
    elif config.convolution_integrator == 'mc':
        return get_monte_carlo