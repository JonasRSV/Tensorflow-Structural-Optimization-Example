import tensorflow as tf
import numpy as np


class VonMises:

    def __init__(self, elasticity_module: float,
                 node_index_matrix: np.ndarray,
                 stretch_freedom: np.ndarray,
                 **kwargs):
        self.elasticity_module = tf.constant(elasticity_module, dtype=tf.float64)
        self.node_index_matrix = tf.constant(node_index_matrix)
        self.stretch_freedom = tf.constant(stretch_freedom, dtype=tf.float64)

    def get_stress_function(self):

        @tf.function
        def stress_function(u: tf.Tensor):
            index_matrix = tf.squeeze(tf.gather(u, self.node_index_matrix))
            stress = tf.matmul(index_matrix, self.stretch_freedom) * self.elasticity_module
            stress = tf.sqrt(
                tf.square(stress[:, 0]) - stress[:, 0] * stress[:, 1] + tf.square(stress[:, 1]) + 3 * tf.square(
                    stress[:, 2]))

            return stress

        return stress_function
