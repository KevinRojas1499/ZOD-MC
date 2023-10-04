import sde_lib

def get_sde(config):

    if config.sde_type == 'vp':
        return sde_lib.SDE(config)
    elif config.sde_type == 've':
        return sde_lib.SDE(config)