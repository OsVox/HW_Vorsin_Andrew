
# важно! все зависимости, которые используете (если добавляее новые) в этом классе надо явно продублировать в эту ячейку
import numpy as np
import random
from scipy.signal import convolve2d

SEED=21
random.seed(SEED)
np.random.seed(SEED)

def game_of_life_next_step(array):
    counter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    to_work = convolve2d(array, counter, mode='same')

    res = np.where(np.logical_or(np.logical_and(array == 0, to_work == 3), 
                           np.logical_and(array == 1, 
                                          np.logical_and(to_work <= 3, to_work >= 2))), 1, 0)
    res[0] = 0
    res[-1] = 0
    res[:, 0] = 0
    res[:, -1] = 0
    return res
