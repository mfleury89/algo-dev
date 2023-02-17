import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError


def gaussian_prob(point, mean, cov):
    try:
        return ((2*np.pi)**(-np.size(point)/2))*((np.linalg.det(cov))**(-1/2)) *\
               np.exp(-(1/2) * np.dot((point - mean), np.dot(np.linalg.inv(cov), (point - mean).T)))
    except LinAlgError:
        return None
    except RuntimeWarning:
        return None


def average_center_movement(current_centers, previous_centers):
    norms = []
    for current_center, previous_center in zip(current_centers, previous_centers):
        norms.append(np.linalg.norm(current_center - previous_center))

    return np.mean(norms)


def em(data, k, function=gaussian_prob, threshold=0.001, iterations=10000, plot=True):
    n_points_ = np.shape(data)[0]
    n_dims = np.shape(data)[1]
    random_indices = np.random.randint(0, n_points_, k)
    current_centers = data[random_indices, :]
    covs = [np.cov(data.T) for _ in range(k)]
    hidden_variables = np.ones((n_points_, k)) / k
    point_classes = np.zeros((n_points_, 1))

    if plot:
        plt.ion()
        fig = plt.figure()
        plot_clusters(data, fig)
    else:
        fig = None

    epsilon = float("inf")
    n_ = 0
    while epsilon > threshold and n_ < iterations:
        for i in range(n_points_):
            point = data[i, :]
            probabilities = []
            normalization_factor = 0
            for center, cov in zip(current_centers, covs):
                prob = function(point, center, cov)
                probabilities.append(prob)
                normalization_factor += prob

            probabilities = [prob / normalization_factor for prob in probabilities]
            hidden_variables[i, :] = np.array(probabilities)
            point_classes[i] = np.argmax(probabilities)

        previous_centers = current_centers.copy()
        for i in range(len(current_centers)):
            current_centers[i] = np.sum(np.expand_dims(hidden_variables[:, i], axis=-1) * data, axis=0) / \
                                 np.sum(hidden_variables[:, i])
            for d in range(n_dims):
                for d_ in range(n_dims):
                    covs[i][d, d_] = np.sum(hidden_variables[:, i] *
                                            (data[:, d] - current_centers[i, d]) *
                                            (data[:, d_] - current_centers[i, d_])) / \
                                            np.sum(hidden_variables[:, i])

        if plot:
            plot_clusters(data, fig, current_centers, point_classes)

        epsilon = average_center_movement(current_centers, previous_centers)
        n_ += 1

    if plot:
        plot_clusters(data, fig, current_centers, point_classes)
        input()

    return current_centers, point_classes


def plot_clusters(points, fig, means=None, colors=None):
    fig.clear()
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    if means is not None:
        plt.scatter(means[:, 0], means[:, 1], marker='D')
    fig.canvas.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    n_points = 10000
    n_dimensions = 2
    n_clusters = 3

    region_size = 1000
    max_variability = 100

    data_ = np.empty((n_points, n_dimensions))
    for j in range(n_clusters - 1):
        m = np.random.uniform(-region_size, region_size)
        s = np.random.uniform(0, max_variability)
        data_[((n_points//n_clusters) * j):((n_points//n_clusters) * (j + 1)), :] = \
            np.random.normal(m, s, (n_points//n_clusters, n_dimensions))

    m = np.random.uniform(-region_size, region_size)
    s = np.random.uniform(0, max_variability)
    n = n_points - (n_clusters - 1) * (n_points//n_clusters)
    data_[(n_clusters - 1) * (n_points//n_clusters):, :] = np.random.normal(m, s, (n, n_dimensions))

    centers, classes = em(data_, k=n_clusters, function=gaussian_prob, threshold=0.001, iterations=10000, plot=True)
