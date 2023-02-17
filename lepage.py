import numpy as np
from itertools import permutations
from scipy.stats import chi2


def probability(t1, t2, sample_size_0, sample_size_1):

    if t1 < 0 or t2 < 0:
        return 0

    def k_(x):
        if x % 2 == 0:
            return x * (x + 2) / 4
        else:
            return ((x + 1)**2) / 4

    m = sample_size_0
    n = sample_size_0 + sample_size_1

    if sample_size_0 == 0:
        if t1 == t2 == 0:
            return 1
        else:
            return 0
    elif sample_size_1 == 0:
        if t1 == m * (m + 1) / 2 and t2 == k_(m):
            return 1
        else:
            return 0
    elif sample_size_0 == 1:
        if n % 2 == 0:
            middle = n / 2
        else:
            middle = (n - 1) / 2

        first_half = np.arange(1, middle + 1)
        last_half = np.arange(middle + 1, n + 1)

        if t1 in first_half and t1 == t2:
            return 1 / n
        elif t1 in last_half and t2 == n + 1 - t1:
            return 1 / n
        else:
            return 0
    elif sample_size_1 == 1:
        if n % 2 == 0:
            middle = n / 2
        else:
            middle = (n - 1) / 2

        first_half_t1 = (n * (n + 1) / 2) - np.arange(1, middle + 1)
        last_half_t1 = (n * (n + 1) / 2) - np.arange(middle + 1, n + 1)
        first_half_t2 = k_(n) - np.arange(1, middle + 1)
        last_half_t2 = k_(n) + np.arange(middle + 1, n + 1) - n - 1

        if t1 in first_half_t1:
            index = np.where(first_half_t1 == t1)[0][0]
            if t2 == first_half_t2[index]:
                return 1 / n
            else:
                return 0
        elif t1 in last_half_t1:
            index = np.where(last_half_t1 == t1)[0][0]
            if t2 == last_half_t2[index]:
                return 1 / n
            else:
                return 0
        else:
            return 0

    prob = ((m * (m - 1)) / (n * (n - 1))) * probability(t1 - 2*m - sample_size_1 + 1, t2 - m, m - 2, sample_size_1) + \
           (m * sample_size_1 / (n * (n - 1))) * probability(t1 - m, t2 - m, m - 1, sample_size_1 - 1) + \
           ((sample_size_1 *
            (sample_size_1 - 1)) / (n * (n - 1))) * probability(t1 - m, t2 - m, m, sample_size_1 - 2) + \
           (m * sample_size_1 / (n * (n - 1))) * probability(t1 - 2*m - sample_size_1 + 1, t2 - m, m - 1,
                                                             sample_size_1 - 1)

    return prob


def lepage_test(sample_0, sample_1, ranking=None, small_samples=False):
    all_samples = np.concatenate((sample_0, sample_1), axis=-1)

    if ranking:
        sort_indices = np.argsort(ranking)
        all_samples = all_samples[sort_indices]
    else:
        all_samples = np.sort(all_samples)

    def compute_t(all_samples_):
        v = np.array([1 if all_samples_[i] in sample_0 else 0 for i in range(np.size(all_samples_))])
        t1 = np.sum(np.multiply(v, np.arange(1, np.size(v) + 1)))
        t2 = (m * (m + n + 1)) / 2 - np.sum(np.multiply(np.abs(np.arange(1, np.size(v) + 1) - (m + n + 1) / 2), v))
        t_stat = ((t1 - mu_1) ** 2) / var_1 + ((t2 - mu_2) ** 2) / var_2
        return t_stat

    m = np.size(sample_0)
    n = np.size(sample_1)

    mu_1 = (m * (m + n + 1)) / 2
    var_1 = (m * n * (m + n + 1)) / 12
    if (m + n) % 2 == 0:
        mu_2 = (m * (m + n + 2)) / 4
        var_2 = m * n * ((m + n) ** 2 - 4) / (48 * (m + n - 1))
    else:
        mu_2 = (m * (m + n + 1) ** 2) / (4 * (m + n))
        var_2 = m * n * (m + n + 1) * ((m + n) ** 2 + 3) / (48 * (m + n) ** 2)

    t = compute_t(all_samples)
    if small_samples:
        sample_permutations = list(permutations(all_samples))
        b = np.size(sample_permutations)
        ts = 0
        for permutation in sample_permutations:
            t_temp = compute_t(permutation)
            if t_temp >= t:
                ts += 1
        p = ts / b
    else:
        p = 1 - chi2.cdf(t, 2)  # asymptotic distribution

    return t, p


if __name__ == '__main__':
    np.random.seed(42)

    s0 = np.random.normal(0, 10, size=4)
    s1 = np.random.normal(100, 20, size=4)
    t_, p_ = lepage_test(s0, s1, small_samples=True)
    print("T value: {}, p value: {}".format(t_, p_))
    t_, p_ = lepage_test(s0, s1, small_samples=False)
    print("T value: {}, p value: {}".format(t_, p_))

    s0 = np.random.normal(0, 1, size=4)
    s1 = np.random.normal(0, 1, size=4)
    t_, p_ = lepage_test(s0, s1, small_samples=True)
    print("T value: {}, p value: {}".format(t_, p_))
    t_, p_ = lepage_test(s0, s1, small_samples=False)
    print("T value: {}, p value: {}".format(t_, p_))

    s0 = np.random.normal(0, 10, size=10000)
    s1 = np.random.normal(100, 20, size=10000)
    t_, p_ = lepage_test(s0, s1, small_samples=False)
    print("T value: {}, p value: {}".format(t_, p_))

    s0 = np.random.normal(0, 1, size=10000)
    s1 = np.random.normal(0, 1, size=10000)
    t_, p_ = lepage_test(s0, s1, small_samples=False)
    print("T value: {}, p value: {}".format(t_, p_))
