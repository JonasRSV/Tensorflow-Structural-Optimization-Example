import tensorflow as tf
import numpy as np


@tf.function
def time_solutions(letter: tf.Tensor, t: tf.Tensor, phi: tf.Tensor):
    broadcast_phi = tf.reshape(phi, [-1, 1])
    broadcasted_sin = tf.sin(t + broadcast_phi) + 1

    return tf.matmul(letter, broadcasted_sin)


@tf.function
def angle_solutions(time_x: tf.Tensor, time_y: tf.Tensor, time_xy: tf.Tensor, theta: tf.Tensor):
    design_variables, steps = tf.shape(time_x)[0], tf.shape(time_x)[1]

    broadcast_time_x = tf.reshape(time_x, [design_variables, steps, 1])
    broadcast_time_y = tf.reshape(time_x, [design_variables, steps, 1])
    broadcast_time_xy = tf.reshape(time_x, [design_variables, steps, 1])

    sigma = (1 / 2) * (broadcast_time_x + broadcast_time_y) + (1 / 2) * (broadcast_time_x - broadcast_time_y) \
            * tf.cos(2 * theta) + broadcast_time_xy * tf.sin(2 * theta)

    tau = (1 / 2) * (broadcast_time_x - broadcast_time_y) * tf.sin(2 * theta) - broadcast_time_xy * tf.cos(2 * theta)

    return sigma, tau


def design_variable_constraint(sigma: tf.Tensor, tau: tf.Tensor, kf: tf.Tensor):
    max_tau = tf.reduce_max(tau, axis=1)
    min_tau = tf.reduce_min(tau, axis=1)
    max_kf = kf * tf.reduce_max(sigma, axis=1)

    condition = (1 / 2) * (max_tau - min_tau) + max_kf

    return tf.reduce_max(condition, axis=1)


@tf.function
def get_stress(strain_vector: tf.Tensor, phis: tf.Tensor, kf: tf.Tensor):
    a = strain_vector[:, 0]
    b = strain_vector[:, 1]
    c = strain_vector[:, 2]

    t = tf.constant(np.linspace(0, 2 * np.pi, 100), dtype=tf.float32)

    time_x = time_solutions(a, t, phis)
    time_y = time_solutions(b, t, phis)
    time_xy = time_solutions(c, t, phis)

    theta = tf.constant(np.linspace(0, np.pi, 100), dtype=tf.float32)
    sigma_solution, tau_solution = angle_solutions(time_x, time_y, time_xy, theta)

    return design_variable_constraint(sigma=sigma_solution, tau=tau_solution, kf=kf)


class Circle:

    def __init__(self, elasticity_module: float,
                 node_index_matrix: np.ndarray,
                 stretch_freedom: np.ndarray,
                 phis: tf.Tensor,
                 kf: tf.Tensor,
                 **kwargs):

        self.elasticity_module = tf.constant(elasticity_module, dtype=tf.float32)
        self.node_index_matrix = tf.constant(node_index_matrix, dtype=tf.int32)
        self.stretch_freedom = tf.constant(stretch_freedom, dtype=tf.float32)
        self.phis = tf.constant(phis, dtype=tf.float32)
        self.kf = tf.constant(kf, dtype=tf.float32)
        self.stress_norm = None

    def get_stress_function(self):

        @tf.function
        def stress_function(u: tf.Tensor):
            n_forces = tf.shape(u)[1]
            n_design = self.node_index_matrix.shape[0]

            offsets = tf.gather(params=u, indices=self.node_index_matrix)
            offsets = tf.reshape(offsets, [n_forces, n_design, 8])

            stress_vector = tf.matmul(offsets, self.stretch_freedom) * self.elasticity_module
            stress_vector = tf.reshape(stress_vector, [n_design, 3, n_forces])

            stress = get_stress(stress_vector, self.phis, self.kf)

            if self.stress_norm is None: self.stress_norm = tf.reduce_max(stress)

            return stress / self.stress_norm

        return stress_function
