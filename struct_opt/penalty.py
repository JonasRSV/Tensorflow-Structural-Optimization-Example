import tensorflow as tf


class NoPenalty:
    def __init__(self, **kwargs):
        self.no_penalty = tf.constant(0.0, dtype=tf.float64)

    def get_penalty_function(self):
        @tf.function
        def penalty_function(sgm_design: tf.Tensor):
            return self.no_penalty


class EntropyPenalty:

    def __init__(self, penalty_size: float, **kwargs):
        self.penalty_size = tf.constant(penalty_size, dtype=tf.float32)

    def get_penalty_function(self):
        @tf.function
        def penalty_function(sgm_design: tf.Tensor):
            entropy = -tf.reduce_sum(sgm_design * tf.math.log(sgm_design)
                                     + (1 - sgm_design) * tf.math.log(1 - sgm_design))

            return self.penalty_size * entropy

        return penalty_function
