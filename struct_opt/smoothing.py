import numpy as np
import tensorflow as tf
from queue import Queue


class NoSmoothing:

    def __init__(self, element_index_matrix, **kwargs):
        pass

    def get_smoothing_function(self):
        @tf.function
        def smoothing_function(design: tf.Tensor):  # no-op
            return design

        return smoothing_function


class GaussianSmoothing:

    def __init__(self, element_index_matrix, smoothing_width=5.0, variance=3.0, **kwargs):
        self.element_index_matrix = element_index_matrix
        self.variance = variance

        n_elements = element_index_matrix.max()
        smoothing_matrix = np.zeros((n_elements + 1, n_elements + 1))

        height, width = element_index_matrix.shape
        for i in range(height):
            for j in range(width):
                if element_index_matrix[i, j] != -1:
                    index = np.array([i, j])

                    current_element_index = element_index_matrix[i, j]
                    elements, densitites = self._expand_from(index, smoothing_width)

                    for element, density in zip(elements, densitites):
                        smoothing_matrix[current_element_index][element] = density

        self.smoothing_matrix = (smoothing_matrix.T / smoothing_matrix.sum(axis=1)).T

        self.smoothing_matrix = tf.constant(self.smoothing_matrix, dtype=tf.float64)

    def _gaussian_kernel(self, x, y):
        return np.exp(-np.square(x - y).sum() / self.variance)

    def _euclidean_distance(self, x, y):
        return np.sqrt(np.square(x - y).sum())

    def __within_bounds(self, index):
        heigth, width = self.element_index_matrix.shape
        if (0 <= index[0] < heigth) and (0 <= index[1] < width) and self.element_index_matrix[index[0]][index[1]] != -1:
            return True
        return False

    def _expand_from(self, index: np.ndarray, width: float):
        elements = []
        densities = []

        origo = np.zeros(2)
        queue = Queue()

        queue.put((origo, index))

        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

        seen_set = set()
        seen_set.add(index.tostring())
        while not queue.empty():
            position, index = queue.get()
            density = self._gaussian_kernel(origo, position)

            elements.append(self.element_index_matrix[index[0]][index[1]])
            densities.append(density)

            for d in directions:
                next_index = index + d
                if self.__within_bounds(next_index) \
                        and next_index.tostring() not in seen_set \
                        and self._euclidean_distance(position, origo) <= width:
                    queue.put((position + d, next_index))
                    seen_set.add(next_index.tostring())

        return elements, densities

    def get_smoothing_function(self):

        @tf.function
        def smoothing_function(design: tf.Tensor):
            return tf.linalg.matvec(self.smoothing_matrix, design)

        return smoothing_function
