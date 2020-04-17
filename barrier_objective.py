import tensorflow as tf


class Barrier():

    def __init__(self, max_constraint: float, barrier_width: float, barrier_size: float):
        self.max_constraint = max_constraint
        self.barrier_width = barrier_width
        self.barrier_size = barrier_size

    def get_objective_function(self):
        @tf.function
        def objective_function(weight: tf.Tensor, constraints: tf.Tensor):
            probabilities = tf.nn.softmax(constraints - tf.reduce_min(constraints))

            barrier = tf.math.log((self.max_constraint - constraints) / self.barrier_size)
            barrier = barrier * probabilities * self.barrier_size

            return weight - tf.reduce_sum(barrier)

        return objective_function
