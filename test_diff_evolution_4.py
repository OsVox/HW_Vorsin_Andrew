import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_4():
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

    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  3.1586555593321464e-09
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  1.5938252939662334e-06
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  5.267711600254188e-09
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  1.2351379012898178e-05
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.0004433015054683409
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  2.34123831432953e-12
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  0.0006425692265228378
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='current')))[-1][1] ==  1.7763568394002505e-14
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  5.329070518200751e-15
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  1.3145040611561853e-13
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  4.696243394164412e-14
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  2.5156543514981422e-12
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  2.3953061756287752e-12
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  1.7663348561569592e-10
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  8.138583140748779e-10
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  3.817779425929757e-11
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  0.0
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection')))[-1][1] ==  9.222862222860833e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current')))[-1][1] ==  0.00016929239603726042
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst')))[-1][1] ==  7.953064645964002e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst')))[-1][1] ==  4.797248952328879e-06
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection')))[-1][1] ==  0.00037558037968324175
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current')))[-1][1] ==  0.0023816855039101303
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst')))[-1][1] ==  2.5684822712800203e-05
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst')))[-1][1] ==  0.005407317348194391
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection')))[-1][1] ==  1.2695345835865277e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='current')))[-1][1] ==  2.8051519111737967e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst')))[-1][1] ==  3.988311496010582e-15
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst')))[-1][1] ==  2.970985647484177e-08
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection')))[-1][1] ==  3.2886353169544126e-09
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current')))[-1][1] ==  4.172793337981942e-07
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst')))[-1][1] ==  3.0160103711769958e-12
    assert np.array(list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst')))[-1][1] ==  8.29268805783664e-08
