import numpy as np


def evaluate(point):
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


def probability(current_val, new_val, temp):
    if new_val >= current_val:
        return 1
    else:
        return np.exp((new_val - current_val) / temp)


def anneal(data, objective_function=evaluate, decay_function=decay, initial_temperature=100, iterations=10000):
    current_point = data[np.random.randint(data.shape[0]), :]
    current_value = objective_function(current_point)
    print("INITIAL POINT")
    print(current_value)
    print(current_point)
    print()
    variability = np.mean(np.linalg.norm(data, axis=-1))
    temperature = initial_temperature
    for i in range(iterations):
        while True:
            new_point = data[np.random.randint(data.shape[0]), :]
            distance = np.linalg.norm(current_point - new_point)
            if distance <= variability:
                break
        new_value = objective_function(new_point)

        p = probability(current_value, new_value, temperature)
        if p == 1:
            current_point = new_point
            current_value = new_value
        else:
            new_choice = [current_point, new_point]
            current_point = new_choice[np.random.choice(2, 1, p=[1 - p, p])[0]]
            current_value = objective_function(current_point)

        temperature = decay_function(temperature, initial_temperature)

    return current_value, current_point


if __name__ == '__main__':
    n_points = 10000
    n_dimensions = 2

    region_size = 1
    max_variability = 100

    m = np.random.uniform(-region_size, region_size)
    s = np.random.uniform(0, max_variability)
    data_ = np.random.normal(m, s, (n_points, n_dimensions))

    max_, point_ = anneal(data_, objective_function=evaluate, decay_function=decay,
                          initial_temperature=1, iterations=1000)
    print("SAMPLED MAX")
    print(max_)
    print(point_)
    print()

    values = map(evaluate, data_)
    print("TRUE MAX:")
    print(max(values))
    print(data_[np.argmax(values), :])