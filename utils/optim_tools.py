import torch.optim as optim
def get_optimizer(config, model):
    return optim.Adam(model.parameters(), lr=config.lr)