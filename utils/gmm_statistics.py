import yaml
import torch

def compute_stats_gmm(data, means):
    N = means.shape[0]
    empirical_means = torch.zeros_like(means)
    num_cluster = torch.zeros((N,1))
    tot = data.shape[0]
    for point in data:
        diffs = torch.sum((means - point)**2,dim=-1)
        idx = torch.argmin(diffs).item()
        print(idx)
        empirical_means[idx] += point
        num_cluster[idx] += 1
    print(empirical_means)
    empirical_means /= num_cluster
    weights = num_cluster/tot
    print(empirical_means.cpu().numpy())
    print(weights.cpu().numpy())
    
    return empirical_means, weights

def get_l2_norm(x):
    return torch.sum(torch.sum(x**2,dim=-1)**.5)

def summarized_stats(config,data):
    params = yaml.safe_load(open(config.density_parameters_path))
    real_means, real_weights = torch.tensor(params['means'],dtype=torch.float32), torch.tensor(params['coeffs'],dtype=torch.float32)
    means, weights = compute_stats_gmm(data, real_means)

    error_means = get_l2_norm(means-real_means)
    error_weights = get_l2_norm(weights-real_weights)
    return error_weights, error_means

