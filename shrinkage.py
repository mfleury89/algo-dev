import numpy as np


def constant_correlation_matrix(samples, covariance=None):
    n = samples.shape[0]
    if covariance is None:
        covariance = np.cov(samples)

    norms = {}
    corr_sum = 0
    for i in range(n - 1):
        norms[i] = {}
        for j in range(i + 1, n):
            norms[i][j] = np.sqrt(covariance[i, i] * covariance[j, j])
            corr_sum += covariance[i, j] / norms[i][j]

    r = corr_sum * 2
    r /= ((n - 1) * n)

    matrix = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = covariance[i, i]
            else:
                matrix[i, j] = matrix[j, i] = r * norms[i][j]

    return matrix, r


def asymmetric_covariance(pair_0, pair_1, residuals, covariance):
    i, j = pair_0
    first_terms = np.multiply(residuals[i, :], residuals[j, :]) - covariance[i, j]

    i, j = pair_1
    second_terms = np.multiply(residuals[i, :], residuals[j, :]) - covariance[i, j]

    return np.mean(np.multiply(first_terms, second_terms))


def apply_shrinkage(samples):
    n = samples.shape[0]
    t = samples.shape[1]
    covariance = np.cov(samples)
    averages = np.expand_dims(np.mean(samples, axis=1), axis=-1)
    residuals = samples - averages

    target, r = constant_correlation_matrix(samples, covariance)

    pi = 0
    rho = 0
    gamma = 0
    for i in range(n):
        for j in range(n):
            pi_sum = asymmetric_covariance((i, j), (i, j), residuals, covariance)
            pi += pi_sum

            if i == j:
                rho += pi_sum

            if i != j:
                rho += (r / 2) * ((np.sqrt(covariance[j, j] / covariance[i, i]) * asymmetric_covariance((i, i), (i, j),
                                                                                                        residuals,
                                                                                                        covariance))
                                  + (np.sqrt(covariance[i, i] / covariance[j, j]) * asymmetric_covariance((j, j),
                                                                                                          (i, j),
                                                                                                          residuals,
                                                                                                          covariance)))

            gamma += np.square(target[i, j] - covariance[i, j])

    kappa = (pi - rho) / gamma
    intensity = max(0, min(kappa / t, 1))

    return intensity * target + (1 - intensity) * covariance


if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.uniform(0, 1, (5, 100))
    outliers = np.random.uniform(1, 2, (5, 10))
    data = np.concatenate((data, outliers), axis=-1)

    shrunk_covariance = apply_shrinkage(data)
    print(np.cov(data))
    print(shrunk_covariance)

    print(np.sum(np.abs(np.cov(data))))
    print(np.sum(np.abs(shrunk_covariance)))
