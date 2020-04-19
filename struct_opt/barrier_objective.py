import tensorflow as tf


class Barrier():

    def __init__(self, max_constraint: float, barrier_width: float, barrier_size: float):
        self.max_constraint = tf.constant(max_constraint, dtype=tf.float32)
        self.barrier_width = tf.constant(barrier_width, dtype=tf.float32)
        self.barrier_size = tf.constant(barrier_size, dtype=tf.float32)

    def get_objective_function(self):
        @tf.function
        def objective_function(weight: tf.Tensor, constraints: tf.Tensor):
            probabilities = tf.nn.softmax(constraints - tf.reduce_min(constraints))

            barrier = tf.math.log((self.max_constraint - constraints) / self.barrier_width)
            barrier = barrier * probabilities * self.barrier_size

            return weight - tf.reduce_sum(barrier)

        return objective_function
