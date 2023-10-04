

def get_beta_function(config):

    def linear_time_change(t):
        return config.multiplier*t + config.bias
    def d_linear_time_change(t):
        return config.multiplier

    if config.beta_func_type == 'linear':
        return linear_time_change, d_linear_time_change