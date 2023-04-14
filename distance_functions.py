import numpy as np

def _cosine_distance(x, w):
    num = (w * x).sum(axis=2)
    denum = np.multiply(np.linalg.norm(w, axis=2), np.linalg.norm(x))
    return 1 - num / (denum+1e-8)

def _euclidean_distance(x, w):
    return np.linalg.norm(np.subtract(x, w), axis=-1)

def _manhattan_distance(x, w):
    return np.linalg.norm(np.subtract(x, w), ord=1, axis=-1)

def _chebyshev_distance(x, w):
    return max(np.subtract(x, w), axis=-1)