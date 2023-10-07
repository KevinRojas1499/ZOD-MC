import sde_lib

def get_sde(config):

    if config.sde_type == 'vp':
        return sde_lib.VP(config)
    elif config.sde_type == 've':
        return sde_lib.VE(config)
    elif config.sde_type == 'edm':
        return sde_lib.EDM(config)