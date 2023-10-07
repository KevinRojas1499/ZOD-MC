import torch

def get_sigma_function(config):

    # VP
    def vp_scheduling(t):
        return (torch.exp(config.multiplier * t**2/2 + config.bias *t) -1)**.5
    # VE
    def ve_scheduling(t):
        return t**.5
    # EDM
    def edm_scheduling(t):
        return t
    
    if config.sde_type == 'vp':
        return vp_scheduling
    elif config.sde_type == 've':
        return ve_scheduling
    elif config.sde_type == 'edm':
        return edm_scheduling
    
def get_scaling_function(config):

    # VP
    def vp_scaling(t):
        return torch.exp(-(config.multiplier * t**2/2 + config.bias * t)/2)
    # VE
    def ve_scaling(t):
        return torch.tensor([1.],device=t.device)
    # EDM
    def edm_scaling(t):
        return torch.tensor([1.],device=t.device)
    
    if config.sde_type == 'vp':
        return vp_scaling
    elif config.sde_type == 've':
        return ve_scaling
    elif config.sde_type == 'edm':
        return edm_scaling