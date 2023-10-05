
# важно! все зависимости, которые используете (если добавляее новые) в этом классе надо явно продублировать в эту ячейку
import numpy as np
import random

SEED = 21
random.seed(SEED)
np.random.seed(SEED)

def find_largest_element(array, n=7):
    array.sort()
    return array[-n:]
