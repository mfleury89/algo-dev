import numpy as np
import matplotlib.pyplot as plt


def average_center_movement(current_centers, previous_centers):
    norms = []
    for current_center, previous_center in zip(current_centers, previous_centers):
        norms.append(np.linalg.norm(current_center - previous_center))

    return np.mean(norms)


def k_means(data, k, threshold=0.001, iterations=10000, plot=True):
    n_points_ = np.shape(data)[0]
    random_indices = np.random.randint(0, n_points_, k)
    current_centers = data[random_indices, :]
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
            distances = []
            for center in current_centers:
                distances.append(np.linalg.norm(point - center))
            point_classes[i] = np.argmin(distances)

        previous_centers = current_centers.copy()
        for i in range(len(current_centers)):
            current_centers[i] = np.mean(data[np.where(point_classes == i)[0], :])

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

    centers, classes = k_means(data_, n_clusters, threshold=0.001, iterations=10000, plot=True)
