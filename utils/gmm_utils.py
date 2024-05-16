import yaml
import torch

def compute_stats_gmm(data, means):
    N = means.shape[0]
    empirical_means = torch.zeros_like(means)
    num_cluster = torch.zeros((N,1))
    tot = 0
    for point in data:
        if torch.sum(torch.isnan(point)) > 0 :
            print("Point is bad")
            continue
        diffs = torch.sum((means - point)**2,dim=-1)
        idx = torch.argmin(diffs).item()
        empirical_means[idx] += point
        num_cluster[idx] += 1
        tot +=1
    empirical_means /= num_cluster
    weights = num_cluster/tot
    # print(empirical_means.cpu().numpy())
    # print(weights.cpu().numpy())
    
    return empirical_means, weights

def compute_stats_gmm_from_config(config, data, device):
    params = yaml.safe_load(open(config.density_parameters_path))
    means = to_tensor(params['means'],device)
    return compute_stats_gmm(data,means)

def get_l2_norm(x):
    return torch.sum(torch.sum(x**2,dim=-1)**.5)

def summarized_stats(config,data):
    params = yaml.safe_load(open(config.density_parameters_path))
    real_means, real_weights = torch.tensor(params['means'],dtype=torch.float32), torch.tensor(params['coeffs'],dtype=torch.float32)
    means, weights = compute_stats_gmm(data, real_means)

    error_means = get_l2_norm(means-real_means)
    error_weights = get_l2_norm(weights-real_weights)
    return error_weights, error_means

def to_tensor(x,device):
    return torch.tensor(x,dtype=torch.float32,device=device)


def sample_from_gmm(config, num_samples,device):
    # Only use this if you dont want to  make a distribution
    from torch.distributions.multivariate_normal import MultivariateNormal
    params = yaml.safe_load(open(config.density_parameters_path))
    c, means, variances = to_tensor(params['coeffs'],device), to_tensor(params['means'],device), to_tensor(params['variances'],device)
    n = len(c)
    d = means[0].shape[0]
    gaussians = [MultivariateNormal(means[i],variances[i]) for i in range(n)]
    samples = torch.zeros(num_samples,d,dtype=torch.float32,device=device)
    for i in range(num_samples):
        idx = torch.randint(0,n, (1,))
        samples[i] = gaussians[idx].sample()
    return samples    