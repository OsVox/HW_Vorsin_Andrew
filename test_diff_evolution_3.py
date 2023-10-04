import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_3():
    from diff_evolution import differential_evolution
    SEED = 21
    random.seed(SEED)
    np.random.seed(SEED)

    def rastrigin(array, A=10):
        return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))
  
    def griewank(array):
        term_1 = (array[0] ** 2 + array[1] ** 2) / 2
        term_2 = np.cos(array[0]/ np.sqrt(2)) * np.cos(array[1]/ np.sqrt(2))
        return 1 + term_1 - term_2
  
    def rosenbrock(array):
        return (1 - array[0]) ** 2 + 100 * (array[1] - array[0] ** 2) ** 2

    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  9.022457447827037e-09
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  4.044373724809702e-10
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  1.7763568394002505e-15
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  2.2562766872624707e-06
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  9.324457650450313e-09
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.0001792338154231743
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  8.189005029635155e-13
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  4.563925637768307e-05
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='current')))[-1][1] ==  1.9539925233402755e-14
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.9949590570932898
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  1.7763568394002505e-15
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  1.7763568394002505e-15
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  5.21249710061511e-13
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  1.8017032310524428e-11
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  7.780442956573097e-13
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  8.940181928096536e-12
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  4.449752566415555e-09
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  4.352338711655079e-09
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  2.2428120991787372e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  6.8610005020808336e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  4.54963851291326e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  9.659293658574029e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.0003583405017178267
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  5.476435354155614e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  3.088107357810844e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  7.681663761139406e-10
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='current')))[-1][1] ==  7.689410044606886e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  3.77437687019181e-14
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  3.035989650540948e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  4.803442283500619e-10
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  2.4542272386157863e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  3.5119405241796733e-13
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  2.316908229883164e-08