import numpy as np


def crop_matrix(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    while matrix.shape[0] > height:
        matrix = np.delete(matrix, -1, axis=0)
    while matrix.shape[1] > width:
        matrix = np.delete(matrix, -1, axis=1)
    return matrix


def padd_matrix(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    result = np.zeros((height, width))
    result[:matrix.shape[0],:matrix.shape[1]] = matrix
    return result


def fit_matrix_size(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    return padd_matrix(crop_matrix(matrix, height, width), height, width)

