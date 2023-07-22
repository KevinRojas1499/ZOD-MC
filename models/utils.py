import torch
from models.cnf import CNF

MODELS = {}


def get_model(name):
    return MODELS[name]

def create_model(config):
    score_model = CNF()
    return score_model