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


def design_variable_constraint(sigma: tf.Tensor, tau: tf.Tensor, kf):
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

    t = tf.constant(np.linspace(0, 2 * np.pi, 100))

    time_x = time_solutions(a, t, phis)
    time_y = time_solutions(b, t, phis)
    time_xy = time_solutions(c, t, phis)

    theta = tf.constant(np.linspace(0, np.pi, 100))
    sigma_solution, tau_solution = angle_solutions(time_x, time_y, time_xy, theta)

    return design_variable_constraint(sigma=sigma_solution, tau=tau_solution, kf=kf)


class Circle:

    def __init__(self, elasticity_module: float,
                 index_matrix: tf.Tensor,
                 stretch_freedom: tf.Tensor,
                 phis: tf.Tensor,
                 kf: tf.Tensor):

        self.elasticity_module = elasticity_module
        self.index_matrix = index_matrix
        self.stretch_freedom = stretch_freedom
        self.phis = phis
        self.kf = kf

    def get_stress_function(self):

        @tf.function
        def stress_function(u: tf.Tensor):
            n_forces = tf.shape(u)[1]
            n_design = self.index_matrix.shape[0]

            index_matrix = tf.reshape(self.index_matrix, [n_forces, n_design, 8])
            strain_vector = tf.matmul(index_matrix, self.stretch_freedom) * self.elasticity_module
            strain_vector = tf.reshape(strain_vector, [n_design, 3, n_forces])
            stress = get_stress(strain_vector, self.phis, self.kf)

            return stress

        return stress_function
