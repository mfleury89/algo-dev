import numpy as np
import matplotlib.pyplot as plt


class Entity:
    def __init__(self, trait_vector):
        self.trait_vector = trait_vector


def fitness(entity, coefficients):
    return np.dot(entity.trait_vector, coefficients)


def select(population, scores):
    # indices = np.where(scores >= np.median(scores))[0]
    indices = np.argsort(scores)[(len(population) // 2):]
    if np.size(indices) % 2 != 0:
        indices = indices[1:]
    return [population[i] for i in indices]


def mutate(population, mutation_rate=0.05):
    for j in range(len(population)):
        for i in range(len(population[j].trait_vector)):
            if np.random.uniform(0, 1) <= mutation_rate:
                population[j].trait_vector[i] = np.abs(population[j].trait_vector[i] - 1)
                # np.random.uniform(0, trait_range)

    return population


def select_breeding_pairs(population):
    breeding_pairs = []
    selected_individuals = []
    while True:
        possible_partners = list(set(range(len(population))) - set(selected_individuals))
        if len(possible_partners) == 0:
            break
        j, k = np.random.choice(possible_partners, 2, replace=False)
        breeding_pairs.append((population[j], population[k]))
        selected_individuals.extend([j, k])

    return breeding_pairs


def uniform_crossover(population):
    breeding_pairs = select_breeding_pairs(population)
    offspring = []
    for pair in breeding_pairs:
        offspring0 = Entity(pair[0].trait_vector.copy())
        offspring1 = Entity(pair[1].trait_vector.copy())

        for i in range(len(offspring0.trait_vector)):
            if np.random.uniform(0, 1) > 0.5:
                offspring0.trait_vector[i] = pair[1].trait_vector[i]
                offspring1.trait_vector[i] = pair[0].trait_vector[i]

        offspring.extend([offspring0, offspring1])

    return offspring


def random_crossover(population):
    breeding_pairs = select_breeding_pairs(population)
    offspring = []
    for pair in breeding_pairs:
        offspring0 = Entity(pair[0].trait_vector.copy())
        offspring1 = Entity(pair[1].trait_vector.copy())

        crossover_point = np.random.randint(0, len(offspring0.trait_vector))
        offspring0.trait_vector[:crossover_point] = pair[1].trait_vector[:crossover_point]
        offspring1.trait_vector[:crossover_point] = pair[0].trait_vector[:crossover_point]

        offspring.extend([offspring0, offspring1])

    return offspring


def genetically_evolve(population, fitness_function, selection_function, mutation_function,
                       replication_function, max_n_generations=10000, convergence_tolerance=0.001, patience=10,
                       coefficients=None):

    n = 0
    previous_max_fitness = 0
    fitness_scores = []
    for entity in population:
        fitness_scores.append(fitness_function(entity, coefficients))
    max_fitness = np.max(fitness_scores)
    all_time_max = max_fitness
    print("GENERATION {} MAX FITNESS: {}".format(n, max_fitness))
    history = [max_fitness]
    count = 0
    while n < max_n_generations:
        population = selection_function(population, fitness_scores)  # select
        if len(population) == 0:
            print("The population has been extinguished.")
            exit()
        offspring = replication_function(population)  # replicate
        population.extend(offspring)
        population = mutation_function(population)  # mutate

        fitness_scores = []
        for entity in population:
            fitness_scores.append(fitness_function(entity, coefficients))

        max_fitness = np.max(fitness_scores)
        if max_fitness > all_time_max:
            all_time_max = max_fitness
        history.append(max_fitness)
        print("GENERATION {} MAX FITNESS: {}".format(n + 1, max_fitness))
        if np.abs(max_fitness - previous_max_fitness) < convergence_tolerance and max_fitness == all_time_max:
            count += 1
            if count == patience:
                return population[np.argmax(fitness_scores)], max_fitness, history
        else:
            count = 0
        previous_max_fitness = max_fitness

        n += 1

    print("Warning: algorithm did not converge, solution may not be optimal.")
    return population[np.argmax(fitness_scores)], max_fitness, history


if __name__ == '__main__':
    population_size = 100
    n_traits = 20
    coeffs_range = 100
    coeffs = np.random.randint(-coeffs_range, coeffs_range, n_traits)
    pop = [Entity(np.random.randint(0, 2, n_traits)) for i in range(population_size)]

    solution, score, hist = genetically_evolve(population=pop, fitness_function=fitness, selection_function=select,
                                               mutation_function=mutate, replication_function=random_crossover,
                                               max_n_generations=1000, convergence_tolerance=0.001, patience=3,
                                               coefficients=coeffs)

    optimal_traits = np.where(coeffs <= 0, 0, 1)
    optimum = np.dot(coeffs, optimal_traits)

    print("SOLUTION: {}".format(solution.trait_vector))
    print("SCORE: {}".format(score))
    print("GLOBAL OPTIMUM: {}".format(optimum))
    plt.plot(hist)
    plt.show()
