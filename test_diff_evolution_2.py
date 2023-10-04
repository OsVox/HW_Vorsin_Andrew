import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_2():
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

    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  7.902087872935226e-10
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  1.2930886263973207e-06
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  4.697851807122788e-06
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  5.7356261962127064e-08
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.00033858769831063285
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  1.580957587066223e-13
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  0.00021997450961919185
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  7.105427357601002e-15
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  1.5987211554602254e-14
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  1.7763568394002505e-15
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  8.626432901337466e-14
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  9.044764937016225e-12
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  5.42512701429132e-11
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  1.529509852105093e-11
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  7.747535946123207e-10
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  7.956492131810933e-10
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  3.960062842304544e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  1.7068164604891756e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  9.345939532357008e-07
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  6.291421812443823e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  0.00037806397695296307
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  7.618182936255569e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  5.341116664177106e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  0.002990757048292397
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  1.5452104130545194e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current')))[-1][1] ==  1.7612832371109777e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  1.2910767433003021e-14
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  1.590405024650817e-10
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  9.220667801561064e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  3.671149271202803e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  4.0295585790760335e-12
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  3.4886110201264687e-10