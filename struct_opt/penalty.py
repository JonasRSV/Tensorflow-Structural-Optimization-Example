import tensorflow as tf


class NoPenalty:
    def __init__(self, **kwargs):
        self.no_penalty = tf.constant(0.0, dtype=tf.float32)

    def get_penalty_function(self):
        @tf.function
        def penalty_function(sgm_design: tf.Tensor):
            return self.no_penalty

        return penalty_function


class EntropyPenalty:

    def __init__(self, penalty_size: float, epochs: float, penalty_epochs: float, **kwargs):
        self.penalty_size = tf.constant(penalty_size, dtype=tf.float32)
        self.entropy_norm = None
        self.epochs = float(epochs)
        self.penalty_epochs = float(penalty_epochs)
        self.epoch = 0.0

    def get_penalty_function(self):
        def penalty_function(sgm_design: tf.Tensor):
            self.epoch += 1
            if self.entropy_norm is None:
                self.entropy_norm = -tf.cast(tf.size(sgm_design), dtype=tf.float32) * 0.5 * tf.math.log(0.5)

            if self.epochs - self.epoch < self.penalty_epochs:
                entropy = -tf.reduce_sum(sgm_design * tf.math.log(sgm_design)
                                         + (1 - sgm_design) * tf.math.log(1 - sgm_design)) / self.entropy_norm

                return self.penalty_size * entropy
            else:
                return 0.0

        return penalty_function


class LinearEntropyPenalty:

    def __init__(self, penalty_size: float, epochs: int, penalty_epochs: int, **kwargs):
        self.penalty_size = tf.constant(penalty_size, dtype=tf.float32)
        self.epochs = float(epochs)
        self.penalty_epochs = float(penalty_epochs)
        self.entropy_norm = None
        self.epoch = 0.0

    def get_penalty_function(self):
        def penalty_function(sgm_design: tf.Tensor):
            self.epoch += 1

            if self.epochs - self.epoch < self.penalty_epochs:
                if self.entropy_norm is None:
                    self.entropy_norm = -tf.cast(tf.size(sgm_design), dtype=tf.float32) * 0.5 * tf.math.log(0.5)

                entropy = -tf.reduce_sum(sgm_design * tf.math.log(sgm_design)
                                         + (1 - sgm_design) * tf.math.log(1 - sgm_design)) / self.entropy_norm

                return self.penalty_size * (self.epoch / self.epochs) * entropy
            else:
                return 0.0

        return penalty_function


class EntropyStructurePenalty:

    def __init__(self, element_index_matrix, penalty_size: float, epochs: int, penalty_epochs: int, **kwargs):
        self.penalty_size = tf.constant(penalty_size, dtype=tf.float32)
        self.epochs = float(epochs)
        self.penalty_epochs = float(penalty_epochs)
        self.entropy_norm = None
        self.epoch = 0.0

    def get_penalty_function(self):
        def penalty_function(sgm_design: tf.Tensor):
            self.epoch += 1

            if self.epochs - self.epoch < self.penalty_epochs:
                if self.entropy_norm is None:
                    self.entropy_norm = -tf.cast(tf.size(sgm_design), dtype=tf.float32) * 0.5 * tf.math.log(0.5)

                entropy = -tf.reduce_sum(sgm_design * tf.math.log(sgm_design)
                                         + (1 - sgm_design) * tf.math.log(1 - sgm_design)) / self.entropy_norm

                return self.penalty_size * (self.epoch / self.epochs) * entropy
            else:
                return 0.0

        return penalty_function
