import tensorflow as tf


class VonMises:

    def __init__(self, elasticity_module: float,
                 index_matrix: tf.Tensor,
                 stretch_freedom: tf.Tensor):
        self.elasticity_module = elasticity_module
        self.index_matrix = index_matrix
        self.stretch_freedom = stretch_freedom

    def get_stress_function(self):

        @tf.function
        def stress_function(u: tf.Tensor):
            index_matrix = tf.squeeze(tf.gather(u, self.index_matrix))
            stress = tf.matmul(index_matrix, self.stretch_freedom) * self.elasticity_module
            stress = tf.sqrt(
                tf.square(stress[:, 0]) - stress[:, 0] * stress[:, 1] + tf.square(stress[:, 1]) + 3 * tf.square(
                    stress[:, 2]))

            return stress

        return stress_function
