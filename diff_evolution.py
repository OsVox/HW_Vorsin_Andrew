
# важно! все зависимости, которые используете (если добавляее новые) в этом классе надо явно продублировать в эту ячейку
import numpy as np
import random
from scipy.stats import qmc
import math

def create_population(dimensions, init_setting, population_size, seed=21):
    if init_setting == 'LatinHypercube':
        population = qmc.LatinHypercube(dimensions, seed=seed)
        assert population.__class__ == qmc.LatinHypercube
    elif init_setting == 'Halton':
        population = qmc.Halton(dimensions, seed=seed)
        assert population.__class__ == qmc.Halton
    elif init_setting == 'Sobol':
        population = qmc.Sobol(dimensions, seed=seed)
        assert population.__class__ == qmc.Sobol
    else:
        population = np.random.rand(population_size, dimensions)
    if not isinstance(population, np.ndarray):
        population = population.random(population_size)
    return population


def rand2_mutant(population, indexes, mutation_coefficient):
    a, b, c, e, d = population[np.random.choice(indexes, 5, replace=False)]
    mutant = a + mutation_coefficient * (d + b - c - e)
    assert 'e' in locals()
    assert 'd' in locals()
    return mutant


def best1_mutant(population, allowed_indexes, mutation_coefficient, best_index):
    b, c = population[
        np.random.choice(allowed_indexes[:best_index] + allowed_indexes[best_index + 1:], 2, replace=False)]
    mutant = population[allowed_indexes[best_index]] + mutation_coefficient * (b - c)
    return mutant


def rand_to_p_best1_mutant(population, allowed_indexes, mutation_coefficient, fitness, p):
    k_th_order = int(p * len(allowed_indexes))
    p_best_index = np.argpartition(fitness[allowed_indexes], k_th_order)[k_th_order]
    p_best = population[allowed_indexes[p_best_index]]

    b, c = population[
        np.random.choice(allowed_indexes[:p_best_index] + allowed_indexes[p_best_index + 1:],
                         size=2, replace=False)]
    return p_best + mutation_coefficient * (b - c)


def differential_evolution(function, bounds, mutation_coefficient=0.5,
                           crossover_coefficient=0.5, population_size=50, iterations=50,
                           init_setting='random', mutation_setting='rand1',
                           selection_setting='current', p_min=0.1, p_max=0.2):
    # Инициализация популяции и получение первичных результатов
    SEED = 21
    random.seed(SEED)
    np.random.seed(SEED)
    dimensions = len(bounds)
    min_bound, max_bound = bounds.T
    diff = np.fabs(max_bound - min_bound)
    population = create_population(dimensions, init_setting, population_size)
    population = min_bound + population * diff
    fitness = np.array([function(value) for value in population])
    # Найти лучший индекс

    best_idx = np.argmin(fitness)
    best = population[best_idx]

    # Цикл эволюции
    for iteration in range(iterations):
        for population_index in range(population_size):  # population_index показывает какой индекс будет заменён
            allowed_indexes = [index for index in range(population_size) if index not in [best_idx, population_index]]
            if mutation_setting == 'rand2':
                mutant = rand2_mutant(population, allowed_indexes, mutation_coefficient)
            elif mutation_setting == 'best1':
                best_from_sample = np.argmin(fitness[allowed_indexes])
                mutant = best1_mutant(population, allowed_indexes, mutation_coefficient, best_from_sample)
            elif mutation_setting == 'rand_to_p_best1':
                p = np.random.uniform(p_min, p_max)
                mutant = rand_to_p_best1_mutant(population, allowed_indexes, mutation_coefficient,
                                                fitness, p)
            else:
                a, b, c = population[np.random.choice(allowed_indexes, 3, replace=False)]
                mutant = a + mutation_coefficient * (b - c)
            mutant = np.clip(mutant, min_bound, max_bound)

            # Оператор кроссовера
            cross_points = np.random.rand(dimensions) < crossover_coefficient
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # Рекомбинация (замена мутантными значениями)
            trial = np.where(cross_points, mutant, population[population_index])
            # Оценка потомка
            result_of_evolution = function(trial)
            # Селекция
            if selection_setting == 'worst':
                selection_index = np.argmax(fitness)
            elif selection_setting == 'random_among_worst':
                possible_to_compare = list(filter(lambda x: fitness[x] > fitness[population_index], allowed_indexes))
                if not possible_to_compare:
                    selection_index = population_index
                else:
                    selection_index = np.random.choice(possible_to_compare, size=1)
            elif selection_setting == 'random_selection':
                selection_index = np.random.choice(allowed_indexes, size=1)
            else:
                selection_index = population_index
            if result_of_evolution < fitness[selection_index]:
                fitness[selection_index] = result_of_evolution
                population[selection_index] = trial
                if result_of_evolution < fitness[best_idx]:
                    best_idx = selection_index
                    best = trial

        yield best, fitness[best_idx]
