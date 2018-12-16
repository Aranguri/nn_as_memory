import numpy as np

def cosine_distance(v1, v2):
    return np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
