import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_1():
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

    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  2.028812673415814e-10
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  1.197346524151044e-07
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  5.329070518200751e-15
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  3.451623165062756e-08
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  6.479763019484608e-07
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.00041887219276759424
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  4.796163466380676e-13
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  0.00043792941555587106
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  6.927791673660977e-14
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  2.4868995751603507e-14
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  5.415138337738767e-10
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  5.0291536127361185e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  3.379606654175286e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  1.623440975814363e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  2.6353992512128422e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  0.0001115965206988795
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.0007829271144532441
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  4.3670919820206185e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  0.0005061350363791974
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  8.781066846358215e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='current')))[-1][1] ==  2.9372280506324224e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  8.947500696426115e-16
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  8.628961748372131e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  2.3673602669422268e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  3.459298758896539e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  1.8467453932420408e-12
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  4.827414434779271e-09