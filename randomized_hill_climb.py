"""
randomized hill climbing: estimated average number of evaluations before reaching the global optimum
"""

import numpy as np
import random


def find_peak(index, f):
    peak = False
    indices_evaluated = []
    while not peak:
        if index == 0:
            f_right = f[index + 1]
            if index not in indices_evaluated:
                indices_evaluated.append(index)
            if index + 1 not in indices_evaluated:
                indices_evaluated.append(index + 1)

            if f_right > f[index]:
                index += 1
            else:
                peak = True

        elif 0 < index < len(f) - 1:
            f_left = f[index - 1]
            f_right = f[index + 1]
            if index not in indices_evaluated:
                indices_evaluated.append(index)
            if index - 1 not in indices_evaluated:
                indices_evaluated.append(index - 1)
            if index + 1 not in indices_evaluated:
                indices_evaluated.append(index + 1)

            if f_left > f[index] and f_right > f[index]:
                direction = random.randint(-1, 1)
                index += direction
            elif f_left > f[index] > f_right:
                index -= 1
            elif f_left < f[index] < f_right:
                index += 1
            else:
                peak = True

        elif index == len(f) - 1:
            f_left = f[index - 1]
            if index not in indices_evaluated:
                indices_evaluated.append(index)
            if index - 1 not in indices_evaluated:
                indices_evaluated.append(index - 1)
            if f_left > f[index]:
                index -= 1
            else:
                peak = True

    return len(indices_evaluated), f[index]


if __name__ == '__main__':
    y1 = list(np.linspace(0, 1, 3))
    y2 = list(np.linspace(1, 0, 2))
    y3 = list(np.linspace(0, 2, 5))
    y4 = list(np.linspace(2, 0, 8))
    y5 = list(np.linspace(0, 4, 4))
    y6 = list(np.linspace(4, 0, 3))
    y7 = list(np.linspace(0, 2, 8))
    y8 = list(np.linspace(2, 0, 2))
    y = y1[:-1] + y2[0:-1] + y3[0:-1] + y4[0:-1] + y5[0:-1] + y6[0:-1] + y7[0:-1] + y8

    samples = []
    iterations = 100000
    for i in range(iterations):
        local_opt = -float("inf")
        total_evals = 0
        while True:
            n = random.randint(1, 28)
            n_evals, local_opt = find_peak(n - 1, y)
            total_evals += n_evals
            if local_opt == max(y):
                print(local_opt)
                break

        samples.append(total_evals)

    print(sum(samples) / iterations)
