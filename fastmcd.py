import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_covariance(data_, fig, ax, sample_mean, sample_cov, mcd_mean, mcd_cov, alpha):
    ax.clear()

    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)

    n_dims_ = data_.shape[1]

    tolerance = np.sqrt(chi2.ppf(1 - alpha, n_dims_))

    ax.scatter(data_[:, 0], data_[:, 1], color='purple')
    confidence_ellipse(sample_mean, sample_cov, ax, n_std=tolerance,
                       label=r'Normal {}% tolerance ellipse'.format((1 - alpha) * 100), edgecolor='pink')
    confidence_ellipse(mcd_mean, mcd_cov, ax, n_std=tolerance,
                       label=r'MCD {}% tolerance ellipse'.format(format((1 - alpha) * 100)),
                       edgecolor='lime', linestyle='-')
    ax.scatter(sample_mean[0], sample_mean[1], c='pink', marker='D')
    ax.scatter(mcd_mean[0], mcd_mean[1], c='lime', marker='D')

    ax.legend()
    fig.canvas.draw()
    plt.pause(0.001)


def confidence_ellipse(mean, cov, ax_, n_std=3.0, facecolor='none', **kwargs):
    if cov.shape[0] > 2 or cov.shape[1] > 2:
        raise ValueError("Data must be two-dimensional to plot")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, linewidth=3,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transform = Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transform + ax_.transData)
    return ax_.add_patch(ellipse)


def mahalanobis_distance(point, mean, inv_cov):
    return np.sqrt(np.dot((point - mean).T, np.dot(inv_cov, point - mean)))


def mcd(sample, h=None, max_iterations=1000, n_initial_subsets=1, fast=False, plot=False, alpha=0.025):

    def c_steps(subset_, n_steps=max_iterations, initial=False):
        previous_mean = np.empty(p)
        previous_cov = np.empty((p, p))
        det_ = None
        i = 0
        while i < n_steps:
            mean_ = np.mean(subset_, axis=0)
            cov_ = np.cov(subset_.T)
            det_ = np.linalg.det(cov)

            if plot:
                plot_covariance(data, fig, ax, sample_mean, sample_cov, mean_, cov_, alpha)

            if np.array_equal(mean_, previous_mean) and np.array_equal(cov_, previous_cov):
                return mean_, cov_, det_

            if det_ == 0:
                return mean_, cov_, det_
            else:
                distances_ = []
                for point_ in sample:
                    distances_.append(mahalanobis_distance(point_, mean_, np.linalg.inv(cov_)))

                distances_ = np.array(distances_)
                indices_ = np.argsort(distances_)
                subset_ = sample[indices_[:h], :]

                previous_mean = mean_
                previous_cov = cov_

            i += 1

        if not initial:
            print("Warning: Algorithm failed to converge: results may not be reliable.  "
                  "Try increasing the maximum number of iterations")

        return previous_mean, previous_cov, det_

    if plot:
        sample_mean = np.mean(data, axis=0)
        sample_cov = np.cov(data.T)
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None
        ax = None
        sample_mean = None
        sample_cov = None

    n = sample.shape[0]
    p = sample.shape[1]
    n_initial_subsets = np.min((n, n_initial_subsets))

    if h is None:
        h = int((np.floor((n + p + 1) / 2) + n) / 2)  # halfway between most robust h and most efficient h
    else:
        if h < np.floor((n + p + 1) / 2) or h > n:
            print("h must be between {} and {}.  Aborting...".format(np.floor((n + p + 1) / 2), n))
            return None, None

    index_sets = []
    dets = []
    for j in range(n_initial_subsets):  # find n_initial_subsets unique subsets of size p + 1
        while True:
            extending = True
            indices = np.random.randint(0, n, size=p + 1)
            subset = sample[indices, :]
            mean = np.mean(subset, axis=0)
            cov = np.cov(subset.T)
            det = np.linalg.det(cov)
            while extending:
                if det == 0:
                    if np.size(indices) == h:
                        return mean, cov

                    indices_left = list(set(range(n)) - set(indices))
                    index = np.random.choice(indices_left)
                    indices = np.concatenate((indices, index))

                    subset = sample[indices, :]
                    mean = np.mean(subset, axis=0)
                    cov = np.cov(subset.T)
                    det = np.linalg.det(cov)
                else:
                    extending = False

            distances = []
            for point in sample:
                distances.append(mahalanobis_distance(point, mean, np.linalg.inv(cov)))

            distances = np.array(distances)
            indices = np.argsort(distances)

            for idxs in index_sets:
                if np.array_equal(indices, idxs):
                    continue
            else:
                index_sets.append(indices)
                break

        if fast:  # if fastmcd, then store determinant to weed out poor performing subsets
            subset = sample[indices[:h], :]
            _, _, det = c_steps(subset, n_steps=2, initial=True)
            dets.append(det)

    if fast:  # if fastmcd, only keep 10 best performing subsets
        min_indices = np.argsort(dets)[:10]
        remaining_subsets = np.array(index_sets)[min_indices]
    else:
        remaining_subsets = index_sets

    results = []
    dets = []
    for indices in remaining_subsets:
        subset = sample[indices[:h]]
        mean, cov, det = c_steps(subset)
        results.append((mean, cov))
        dets.append(det)

    min_index = int(np.argmin(dets))
    mean, cov = results[min_index]

    if plot:
        plot_covariance(data, fig, ax, sample_mean, sample_cov, mean, cov, alpha)
        input()

    return mean, cov


if __name__ == '__main__':
    alpha_ = 0.025
    n_dims = 2
    n_points = 1000
    fraction_outliers = 0.2

    data = np.random.normal(0, 100, size=(int((1 - fraction_outliers) * n_points), n_dims))
    data_outliers = np.random.uniform(500, 1000, size=(int(fraction_outliers * n_points), n_dims))
    data = np.concatenate((data, data_outliers), axis=0)

    m, c = mcd(data, n_initial_subsets=1, fast=False, plot=True, alpha=alpha_)
