import numpy as np
import matplotlib.pyplot as plt


def compute_cost(data, cluster_vector):
    costs = []
    for cluster in np.unique(cluster_vector):
        indices = np.where(cluster_vector == cluster)[0]
        centroid = np.mean(data[indices, :], axis=0)
        cost = np.sum(np.abs(data[indices, :] - centroid)**2)
        costs.append(cost)

    return np.sum(costs)


def sample_cluster_vector(prob_matrix):
    c = prob_matrix.cumsum(axis=1)  # vectorized way of selecting n_points_ cluster values with n_points_
    u = np.random.rand(len(c), 1)   # probability_matrix rows
    cluster_vector = (u < c).argmax(axis=1)
    return cluster_vector


def find_quantile(scores, quantile):
    scores = np.sort(scores)
    quantile_index = int(np.ceil(quantile * np.size(scores)))
    return scores[quantile_index]


def cross_entropy_clustering(data, n_clusters_, sample_size=1000, quantile=0.1, stop=3, max_iterations=10000,
                             plot=True):
    n_points_ = np.shape(data)[0]
    probability_matrix = np.ones((n_points_, n_clusters_))
    probability_matrix /= n_clusters_

    if plot:
        plt.ion()
        fig = plt.figure()
        plot_clusters(data, fig)
    else:
        fig = None

    quantile_buffer = []
    history_ = {'quantiles': [], 'probabilities': [], 'min_costs': []}
    for _ in range(max_iterations):
        costs = []
        cluster_vectors = []
        for _ in range(sample_size):
            cluster_vector = sample_cluster_vector(probability_matrix)
            cluster_vectors.append(cluster_vector)

            cost = compute_cost(data, cluster_vector)
            costs.append(cost)

        history_['min_costs'].append(np.min(costs))
        if plot:  # plot best performing clustering from current sample
            max_index = np.argmax(costs)
            plot_clusters(data, fig, np.array(cluster_vectors)[max_index])

        quantile_cost = find_quantile(costs, quantile)
        history_['quantiles'].append(quantile_cost)
        quantile_buffer.append(quantile_cost)
        if len(quantile_buffer) > stop:
            del quantile_buffer[0]

        for i in range(n_points_):
            for j in range(n_clusters_ - 1):
                low_cost_indicator = np.where(costs <= quantile_cost, 1, 0)
                low_cost_indicator_indices = np.where(costs <= quantile_cost)[0]

                numerator = 0
                for index in low_cost_indicator_indices:
                    cluster_indicator = int(cluster_vectors[index][i] == j)
                    numerator += low_cost_indicator[index] * cluster_indicator

                probability_matrix[i, j] = numerator / np.sum(low_cost_indicator)
            probability_matrix[i, -1] = 1 - np.sum(probability_matrix[i, :-1])

        history_['probabilities'].append(np.copy(probability_matrix))

        quantile_buffer_indicator = np.where(quantile_buffer == quantile_buffer[-1], 1, 0)
        if np.sum(quantile_buffer_indicator) >= stop:
            break

    cluster_vector = sample_cluster_vector(probability_matrix)

    if plot:  # plot final clustering from final probability matrix
        plot_clusters(data, fig, cluster_vector)

    return cluster_vector, history_


def plot_clusters(points, fig, colors=None):
    fig.clear()
    means = []
    for j in np.unique(colors):
        indices = np.where(colors == j)[0]
        means.append(np.mean(points[indices, :], axis=0))

    means = np.array(means)
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    if means is not None:
        plt.scatter(means[:, 0], means[:, 1], marker='D')
    fig.canvas.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    n_points = 1000
    n_dimensions = 2
    n_clusters = 3

    region_size = 1000
    max_variability = 100

    cutoff = 0.1

    data_ = np.empty((n_points, n_dimensions))
    for j_ in range(n_clusters - 1):
        m = np.random.uniform(-region_size, region_size)
        s = np.random.uniform(0, max_variability)
        data_[((n_points//n_clusters) * j_):((n_points//n_clusters) * (j_ + 1)), :] = \
            np.random.normal(m, s, (n_points//n_clusters, n_dimensions))

    m = np.random.uniform(-region_size, region_size)
    s = np.random.uniform(0, max_variability)
    n = n_points - (n_clusters - 1) * (n_points//n_clusters)
    data_[(n_clusters - 1) * (n_points//n_clusters):, :] = np.random.normal(m, s, (n, n_dimensions))

    clustering, history = cross_entropy_clustering(data_, n_clusters, sample_size=1000, quantile=cutoff, stop=3,
                                                   max_iterations=10000, plot=True)

    plt.ioff()
    fig_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    ax1.set_title('{} Quantile per Sample'.format(cutoff))
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Cost')
    ax1.plot(history['quantiles'])

    ax2.set_title('Minimum Cost per Sample')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Cost')
    ax2.plot(history['min_costs'])
    plt.show()
