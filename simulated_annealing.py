import sys
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

stdhandler = logging.StreamHandler(sys.stdout)
filehandler = logging.FileHandler('{}.log'.format(__name__))

formatter = logging.Formatter('%(asctime)-15s %(name)s - %(levelname)s - %(message)s')
stdhandler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.handlers = [stdhandler, filehandler]


def evaluate(point, **kwargs):
    return -np.linalg.norm(point)


def decay(temp, max_temp):
    min_temp = 1
    space = np.linspace(max_temp, min_temp)
    length = np.size(space)
    index = np.where(space == temp)[0][0]
    if index + 1 < length:
        return space[index + 1]
    else:
        return min_temp


def no_decay(temp, _):
    return temp


def probability(current_val, new_val, target, temp):
    if abs(new_val - target) <= abs(current_val - target):
        return 1
    else:
        return np.min((np.exp(-abs(new_val - current_val) / temp), 1))


def sample(space, point=None, step_width=1):
    new_point = []
    for idx, bounds in enumerate(space):
        if point is not None:
            width = abs(bounds[1] - bounds[0]) * step_width
            new_point.append(
                np.random.uniform(np.max((bounds[0], point[idx] - width)), np.min((bounds[1], point[idx] + width))))
        else:
            new_point.append(np.random.uniform(bounds[0], bounds[1]))

    return np.array(new_point)


def anneal(space, objective_function, guess=None, target=0, tolerance=1e-3, sample_function=sample,
           probability_function=probability, decay_function=decay, initial_temperature=100, iterations=10000, **kwargs):
    if guess is not None:
        current_point = guess
    else:
        current_point = sample_function(space)
    current_value = objective_function(current_point, **kwargs)
    best_point = current_point
    best_value = current_value

    if abs(best_value - target) <= tolerance:
        logger.info("Initial point minimizes objective.  Point: %s, value: %s", best_point, best_value)
        return best_value, best_point

    logger.info("Initial point: %s, value: %s", current_point, current_value)

    variability = np.mean(np.linalg.norm(space, axis=-1))
    temperature = initial_temperature
    for i in range(iterations):
        while True:
            new_point = sample_function(space, current_point, **kwargs)
            distance = np.linalg.norm(current_point - new_point)
            if distance <= variability:
                break
        new_value = objective_function(new_point, **kwargs)

        p = probability_function(current_value, new_value, target, temperature)
        if p == 1:
            current_point = new_point
            current_value = new_value
        else:
            new_choice = [current_point, new_point]
            new_val = [current_value, new_value]
            idx = np.random.choice(2, 1, p=[1 - p, p])[0]
            current_point = new_choice[idx]
            current_value = new_val[idx]

        # logger.info("Current point: %s, value: %s", current_point, current_value)

        if abs(current_value - target) < abs(best_value - target):
            best_point = current_point
            best_value = current_value

        if abs(best_value - target) <= tolerance:
            break

        temperature = decay_function(temperature, initial_temperature)

    logger.info("Best point: %s, value: %s", best_point, best_value)

    return best_point, best_value


if __name__ == '__main__':
    n_points = 10000
    n_dimensions = 2
    max_size = 10000

    region_sizes = [np.random.uniform(1, max_size) for _ in range(n_dimensions)]
    spaces = [(-region_size, region_size) for region_size in region_sizes]

    point_, max_ = anneal(spaces, objective_function=evaluate, guess=np.array([1e-3, 1e-3]),
                          initial_temperature=int(1e5), iterations=10000, step_width=1e-7)

    logger.info("True max: %s, %s", [0 for _ in range(n_dimensions)], 0)
