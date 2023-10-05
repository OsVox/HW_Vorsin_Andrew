
# важно! все зависимости, которые используете (если добавляее новые) в этом классе надо явно продублировать в эту ячейку
import numpy as np
import random

SEED = 21
random.seed(SEED)
np.random.seed(SEED)

def task_00(A):
    return np.einsum('i->', A)

def task_01(A, B):
    return np.einsum('i,i->i', A, B)

def task_02(A, B):
    return np.einsum('i,i->', A, B)

def task_03(A, B):
    return np.einsum('i,j->ij', A, B)
