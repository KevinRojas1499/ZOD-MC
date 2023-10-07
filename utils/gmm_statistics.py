import numpy as np

def compute_stats_gmm(data):
    limit = 3
    bias = 0
    clusters = [ [] for i in range(5)]
    # center, up right, up left , down left, down right
    for point in data:
        x,y = point
        point = point.numpy()
        if x > limit + bias and y > limit + bias:
            clusters[1].append(point)
        elif x < -limit + bias and y > limit + bias:
            clusters[2].append(point)
        elif x < -limit + bias and y < -limit + bias:
            clusters[3].append(point)
        elif x > limit + bias and y < -limit + bias:
            clusters[4].append(point)
        else:
            clusters[0].append(point)

    stats_x = {"center":0,"up right":0, "up left":0, "down left": 0, "down right":0}
    stats_y = {"center":0,"up right":0, "up left":0, "down left": 0, "down right":0}
    weights = {"center":0,"up right":0, "up left":0, "down left": 0, "down right":0}

    for i, (key, value) in enumerate((stats_x.items())):
        mean = np.mean(np.array(clusters[i]),axis=0) 
        stats_x[key] = mean[0]
        stats_y[key] = mean[1]
        weights[key] = len(clusters[i])/len(data)
    
    return stats_x, stats_y, weights

def to_np_array(data):
    np_data = np.zeros(len(data))
    for i, (key, value) in enumerate((data.items())):
        np_data[i] = value
    
    return np_data

def summarized_stats(data):
    stats_x, stats_y, weights = compute_stats_gmm(data)
    weights = to_np_array(weights)
    real_weights = np.array([.2,.2,.2,.2,.2])
    w = np.sum((weights-real_weights)**2)**.5

    means_x, means_y = np.expand_dims(to_np_array(stats_x),axis=1), np.expand_dims(to_np_array(stats_y),axis=1)
    m = 0
    b = 5
    real_means = np.array([[m,m],[m+b,m+b],[m-b,m+b],[m-b, m -b],[m + b,m - b]])
    means = np.concatenate((means_x,means_y), axis=1)
    error_means = 0
    for i, mean in enumerate(real_means):
        error_means += np.sum((real_means[i]-means[i])**2)**.5

    return w, error_means

