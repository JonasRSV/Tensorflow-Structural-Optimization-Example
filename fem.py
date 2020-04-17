import tensorflow as tf


class FEM:

    def __init__(self, index_vector: tf.Tensor,
                 F: tf.Tensor,
                 element_stiffness: tf.Tensor,
                 freedom_indexes: tf.Tensor,
                 k_dim: tf.Tensor,
                 elasticity_module: float):
        self.index_vector = index_vector
        self.F = F
        self.element_stiffness = element_stiffness
        self.freedom_indexes = freedom_indexes
        self.k_dim = k_dim
        self.elasticity_module = elasticity_module

    def get_fem_function(self):

        def fem_function(design: tf.Tensor):
            """ Create K"""
            element_stiffness = self.element_stiffness * self.elasticity_module
            element_stiffness = tf.reshape(element_stiffness, [1, 64])
            design = tf.reshape(design, [-1, 1])

            value_matrix = tf.matmul(design, element_stiffness)
            value_vector = tf.reshape(value_matrix, [-1])

            """ Sum duplicate coordinates"""
            linearized = tf.matmul(self.index_vector, [[self.k_dim], [1]])
            y, idx = tf.unique(tf.squeeze(linearized))
            value_vector = tf.math.unsorted_segment_sum(value_vector, idx, tf.size(y))  # Here is sum
            y = tf.expand_dims(y, 1)
            index_vector = tf.concat([y % self.k_dim, y // self.k_dim], axis=1)

            """Set values of K"""
            sparse = tf.SparseTensor(indices=index_vector, values=value_vector, dense_shape=[self.k_dim, self.k_dim])
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

