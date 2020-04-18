import tensorflow as tf
import numpy as np


class FEM:

    def __init__(self, node_index_vector: np.ndarray,
                 F: np.ndarray,
                 element_stiffness: np.ndarray,
                 freedom_indexes: np.ndarray,
                 k_dim: np.ndarray,
                 elasticity_module: float):
        self.node_index_vector = tf.constant(node_index_vector, dtype=tf.int64)
        self.F = tf.constant(F, dtype=tf.float32)
        self.element_stiffness = tf.constant(element_stiffness, dtype=tf.float32)
        self.freedom_indexes = tf.constant(freedom_indexes, dtype=tf.int32)
        self.k_dim = tf.constant(k_dim, dtype=tf.int64)
        self.elasticity_module = tf.constant(elasticity_module, dtype=tf.float32)

    def get_fem_function(self):

        def fem_function(design: tf.Tensor):
            """ Create K"""
            element_stiffness = self.element_stiffness * self.elasticity_module
            element_stiffness = tf.reshape(element_stiffness, [1, 64])
            design = tf.reshape(design, [-1, 1])

            value_matrix = tf.matmul(design, element_stiffness)
            value_vector = tf.reshape(value_matrix, [-1])

            """ Sum duplicate coordinates"""
            linearized = tf.matmul(self.node_index_vector, [[self.k_dim], [1]])
            y, idx = tf.unique(tf.squeeze(linearized))
            value_vector = tf.math.unsorted_segment_sum(value_vector, idx, tf.size(y))  # Here is sum
            y = tf.expand_dims(y, 1)
            node_index_vector = tf.concat([y % self.k_dim, y // self.k_dim], axis=1)

            """Set values of K"""
            sparse = tf.SparseTensor(indices=node_index_vector, values=value_vector, dense_shape=[self.k_dim, self.k_dim])
            sparse = tf.sparse.reorder(sparse)
            K = tf.sparse.to_dense(sparse, default_value=0.0)

            """Get small K and F"""
            K_new = tf.transpose(tf.gather(K, self.freedom_indexes))
            K_new = tf.transpose(tf.gather(K_new, self.freedom_indexes))
            F_new = tf.gather(self.F, self.freedom_indexes)

            """fem"""
            inverse = tf.linalg.inv(K_new)
            u = tf.linalg.matmul(inverse, F_new)

            freedom_indexes = tf.expand_dims(self.freedom_indexes, axis=1)
            """Scatter up to big U"""
            u = tf.scatter_nd(freedom_indexes, u, shape=[self.k_dim, self.F.shape[1]])

            return u

        return fem_function

