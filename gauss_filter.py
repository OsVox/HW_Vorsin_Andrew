
# важно! все зависимости, которые используете (если добавляее новые) в этом классе надо явно продублировать в эту ячейку
import numpy as np

def gauss_filter(sigma=1.0, mu=0.0):
    X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    X = np.sqrt(X ** 2 + Y ** 2) - mu
    
    return np.exp(-X**2 / (2 * sigma ** 2))
