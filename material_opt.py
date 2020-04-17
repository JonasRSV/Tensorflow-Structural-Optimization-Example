import tensorflow as tf
import time
import numpy as np
import sys

ELASTICITY_MODULE = 3000
KF = 0.3

CONSTRAINT_VALUE = tf.constant(3000.0, dtype=tf.float64)


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


def design_variable_constraint(sigma: tf.Tensor, tau: tf.Tensor):
    max_tau = tf.reduce_max(tau, axis=1)
    min_tau = tf.reduce_min(tau, axis=1)
    max_kf = KF * tf.reduce_max(sigma, axis=1)

    condition = (1 / 2) * (max_tau - min_tau) + max_kf

    return tf.reduce_max(condition, axis=1)


@tf.function
def get_strain(strain_vector: tf.Tensor, phis: tf.Tensor):
    a = strain_vector[:, 0]
    b = strain_vector[:, 1]
    c = strain_vector[:, 2]

    t = tf.constant(np.linspace(0, 2 * np.pi, 100))

    time_x = time_solutions(a, t, phis)
    time_y = time_solutions(b, t, phis)
    time_xy = time_solutions(c, t, phis)

    theta = tf.constant(np.linspace(0, np.pi, 100))
    sigma_solution, tau_solution = angle_solutions(time_x, time_y, time_xy, theta)

    return design_variable_constraint(sigma=sigma_solution, tau=tau_solution)


@tf.function
def weakest_link(u: tf.Tensor,
                 index_matrix: tf.Tensor,
                 stretch_freedom: tf.Tensor,
                 phis: tf.Tensor):
    #n_forces = tf.shape(u)[1]
    #n_design = index_matrix.shape[0]

    index_matrix = tf.squeeze(tf.gather(u, index_matrix))
    stress = tf.matmul(index_matrix, stretch_freedom) * ELASTICITY_MODULE
    stress = tf.sqrt(tf.square(stress[:, 0]) - stress[:, 0] * stress[:, 1] + tf.square(stress[:, 1]) + 3 * tf.square(stress[:, 2]))
    # index_matrix = tf.reshape(index_matrix, [n_forces, n_design, 8])

    # strain_vector = tf.matmul(index_matrix, stretch_freedom) * ELASTICITY_MODULE
    # strain_vector = tf.reshape(strain_vector, [n_design, 3, n_forces])
    # strain = get_strain(strain_vector, phis)

    return stress


@tf.function
def constraint(design: tf.Tensor,
               index_matrix: tf.Tensor,
               index_vector: tf.Tensor,
               F: tf.Tensor,
               phis: tf.Tensor,
               stretch_freedom: tf.Tensor,
               element_stiffness: tf.Tensor,
               freedom_indexes: tf.Tensor,
               k_dim: tf.Tensor):

    #design = tf.ones(shape=design.shape, dtype=tf.float64)

    """ Create K"""
    element_stiffness = element_stiffness * ELASTICITY_MODULE
    element_stiffness = tf.reshape(element_stiffness, [1, 64])
    design = tf.reshape(design, [-1, 1])

    value_matrix = tf.matmul(design, element_stiffness)
    value_vector = tf.reshape(value_matrix, [-1])

    """ Sum duplicate coordinates"""
    linearized = tf.matmul(index_vector, [[k_dim], [1]])
    y, idx = tf.unique(tf.squeeze(linearized))
    value_vector = tf.math.unsorted_segment_sum(value_vector, idx, tf.size(y)) # Here is sum
    y = tf.expand_dims(y, 1)
    index_vector = tf.concat([y % k_dim, y // k_dim], axis=1)

    """Set values of K"""
    sparse = tf.SparseTensor(indices=index_vector, values=value_vector, dense_shape=[k_dim, k_dim])
    sparse = tf.sparse.reorder(sparse)
    K = tf.sparse.to_dense(sparse, default_value=0.0)

    """Get small K and F"""
    K_new = tf.transpose(tf.gather(K, freedom_indexes))
    K_new = tf.transpose(tf.gather(K_new, freedom_indexes))
    F_new = tf.gather(F, freedom_indexes)


    """fem"""
    inverse = tf.linalg.inv(K_new)
    u = tf.linalg.matmul(inverse, F_new)

    freedom_indexes = tf.expand_dims(freedom_indexes, axis=1)
    """Scatter up to big U"""
    u = tf.scatter_nd(freedom_indexes, u, shape=[k_dim, F.shape[1]])

    return weakest_link(u,
                        index_matrix=index_matrix,
                        stretch_freedom=stretch_freedom,
                        phis=phis)


@tf.function
def weight(design: tf.Tensor):
    return tf.reduce_sum(design)


@tf.function
def objective(design: tf.Variable,
              smoothing_matrix: tf.Tensor,
              index_matrix: tf.Tensor,
              index_vector: tf.Tensor,
              F: tf.Tensor,
              phis: tf.Tensor,
              stretch_freedom: tf.Tensor,
              element_stiffness: tf.Tensor,
              freedom_indexes: tf.Tensor,
              k_dim: tf.Tensor,
              smoothing: tf.Tensor):
    design = tf.sigmoid(design)

    # design = tf.linalg.matvec(smoothing_matrix, design)

    w = weight(design)
    c = constraint(design=design,
                   index_matrix=index_matrix,
                   index_vector=index_vector,
                   F=F,
                   phis=phis,
                   stretch_freedom=stretch_freedom,
                   element_stiffness=element_stiffness,
                   freedom_indexes=freedom_indexes,
                   k_dim=k_dim)

    w_obj = w
    c_obj = smoothing * tf.reduce_mean(tf.abs(tf.math.log(tf.maximum((CONSTRAINT_VALUE - c) / smoothing, 0.00000001))))

    return w_obj + c_obj, w, tf.reduce_max(c), c


def make_smoothing(n_design: int):
    return tf.constant(np.random.rand(n_design, n_design) / n_design)


def make_F(bracket: int,
           tiny_bracket: int,
           index_matrix: np.ndarray,
           elements: np.ndarray,
           directions: np.ndarray,
           amplitudes: np.ndarray):
    force_size = 2 * (tiny_bracket + 1) * (2 * bracket - tiny_bracket + 1)

    F = np.zeros((force_size, amplitudes.size))
    for i in range(elements.size):
        row = index_matrix[elements[i]]

        if directions[i] == 0:
            positions = row[0::2]
        else:
            positions = row[1::2]

        for position in positions:
            F[position, i] = amplitudes[i] / len(positions)

    return tf.constant(F)


def get_index_vector(index_matrix: np.ndarray):
    column_index_for_K = np.kron(index_matrix.T, np.ones(8)).astype(np.int).T.reshape(-1, 1)
    row_index_for_K = np.kron(index_matrix, np.ones(8)).astype(np.int).reshape(-1, 1)

    index_vector = np.concatenate([column_index_for_K, row_index_for_K], axis=1)

    return index_vector


def setup_strain_problem(bracket_heigth: int,
                         tiny_bracket_width: int,
                         element_length: float,
                         thickness: float,
                         poisson_ratio: float):
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])

    A = np.vstack([np.hstack([A11, A12]), np.hstack([A12.T, A11])])
    B = np.vstack([np.hstack([B11, B12]), np.hstack([B12.T, B11])])

    element_stiffness = (A + poisson_ratio * B) * thickness / ((1 - np.square(poisson_ratio)) * 24)

    stretch_matrix = np.array([[-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                               [0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0],
                               [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0]])

    stretch_matrix = stretch_matrix * 1 / (2 * element_length)

    freedom_matrix = np.array([[1.0, poisson_ratio, 0.0],
                               [poisson_ratio, 1.0, 0.0],
                               [0.0, 0.0, (1 - poisson_ratio) / 2]])

    freedom_matrix = (1 / (1 - np.square(poisson_ratio))) * freedom_matrix

    stretch_freedom = freedom_matrix @ stretch_matrix

    def make_index_matrix(heigth: int, width: int):
        n_nodes_heigth = heigth + 1
        n_nodes_width = width + 1

        node_indexes = np.arange(0, n_nodes_width * n_nodes_heigth) + 1
        node_indexes = node_indexes.reshape(n_nodes_width, n_nodes_heigth).T

        strange_matrix = 2 * node_indexes[0:-1, 0:-1] + 1
        strange_matrix = strange_matrix.T.flatten().reshape(-1, 1)
        strange_matrix = np.concatenate([strange_matrix for _ in range(8)], axis=1)

        sub_element = 2 * heigth + np.array([2, 3, 0, 1])
        first_element_indexes = np.array([0, 1, *sub_element, -2, -1]).reshape(-1, 1)

        element_indexes = np.concatenate([first_element_indexes for _ in range(width * heigth)], axis=1).T

        return element_indexes + strange_matrix

    remaining_elements = bracket_heigth - tiny_bracket_width
    remaining_height = tiny_bracket_width
    remaining_width = remaining_elements

    index_matrix_top = make_index_matrix(bracket_heigth, tiny_bracket_width)
    index_matrix_bottom = make_index_matrix(remaining_height, remaining_width)

    index_matrix_bottom = index_matrix_bottom + 2 * (tiny_bracket_width + 1) * bracket_heigth
    index_matrix = np.concatenate([index_matrix_top, index_matrix_bottom], axis=0)

    return index_matrix - 1, \
           tf.constant(stretch_freedom.T), \
           tf.constant(element_stiffness)


def make_fixed_degrees_of_freedom(bracket: int, tiny_bracket: int, k_dim: int):
    tiny_indexes = np.arange(0, tiny_bracket + 1)
    tiny_indexes = tiny_indexes * 2 * (bracket + 1)

    locked_indexes = tiny_indexes + 1

    fixed_degrees_of_freedom = np.concatenate([tiny_indexes, locked_indexes])
    all_values = np.arange(0, 2 * (tiny_bracket + 1) * (2 * bracket - tiny_bracket + 1))

    freedom_indexes = np.array([index for index in all_values if index not in fixed_degrees_of_freedom])

    return freedom_indexes


def initialize_env(super_constant: int,
                   thickness: float,
                   poisson_ratio: float,
                   radius: int,
                   elements: np.ndarray,
                   directions: np.ndarray,
                   amplitudes: np.ndarray,
                   initial_values_design=1.0):
    timestamp = time.time()

    bracket = super_constant * 5
    tiny_bracket = super_constant * 2
    element_length = 1 / bracket
    n_design = np.square(super_constant) * 16

    """Vector to optimise"""
    design_variables = tf.Variable(np.ones(n_design) * initial_values_design, trainable=True)

    """Smoothing matrix for the design variables"""
    smoothing_matrix = make_smoothing(n_design)

    """Constant matrices for pre and post processing in the solver"""
    index_matrix, stretch_freedom, element_stiffness = setup_strain_problem(
        bracket_heigth=bracket,
        tiny_bracket_width=tiny_bracket,
        element_length=element_length,
        thickness=thickness,
        poisson_ratio=poisson_ratio
    )

    """Vector for constructing K matrix"""
    index_vector = get_index_vector(index_matrix=index_matrix)

    """Force vector for FEM"""
    F = make_F(bracket=bracket,
               tiny_bracket=tiny_bracket,
               index_matrix=index_matrix,
               elements=elements,
               directions=directions,
               amplitudes=amplitudes)

    """Dimension of K matrix"""
    k_dim = tf.constant(index_matrix.max(), dtype=tf.int64) + 1

    """Fixed degrees of freedom mask"""
    freedom_indexes = make_fixed_degrees_of_freedom(bracket=bracket,
                                                       tiny_bracket=tiny_bracket,
                                                       k_dim=k_dim)

    print(f"Initializing env: {time.time() - timestamp} seconds")

    return design_variables, smoothing_matrix, tf.constant(index_matrix), \
           tf.constant(index_vector), F, stretch_freedom, element_stiffness, \
           tf.constant(freedom_indexes), k_dim


def train_op(design: tf.Variable,
             smoothing_matrix: tf.Tensor,
             index_matrix: tf.Tensor,
             index_vector: tf.Tensor,
             F: tf.Tensor,
             phis: tf.Tensor,
             stretch_freedom: tf.Tensor,
             element_stiffness: tf.Tensor,
             freedom_indexes: tf.Tensor,
             k_dim: tf.Tensor,
             optimizer: tf.keras.optimizers.Optimizer,
             smoothing: float = 100):
    smoothing = tf.constant(smoothing, dtype=tf.float64)

    with tf.GradientTape() as tape:
        obj, w, c, all_c = objective(design,
                                     smoothing_matrix=smoothing_matrix,
                                     index_matrix=index_matrix,
                                     index_vector=index_vector,
                                     F=F,
                                     phis=phis,
                                     stretch_freedom=stretch_freedom,
                                     k_dim=k_dim,
                                     element_stiffness=element_stiffness,
                                     freedom_indexes=freedom_indexes,
                                     smoothing=smoothing)

    gradients = tape.gradient(obj, design)
    optimizer.apply_gradients([(gradients, design)])

    return obj, w, c, all_c


if __name__ == "__main__":
    super_constant = 5

    design_variables, smoothing_matrix, index_matrix, index_vector, \
    F, stretch_freedom, element_stiffness, freedom_indexes, k_dim = \
        initialize_env(super_constant=super_constant,
                       thickness=0.02,
                       poisson_ratio=0.3,
                       radius=1.0,
                       elements=np.array([
                           16 * np.square(super_constant) - super_constant * 2
                       ]),
                       directions=np.array([
                           1
                       ]),
                       amplitudes=np.array([
                           -1
                       ]),
                       initial_values_design=2.0)

    phis = tf.constant([0.0], dtype=tf.float64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    smoothing = 1000

    constraints, weights, designs, all_constraints = [], [], [design_variables.numpy()], []
    for e in range(100):
        timestamp = time.time()
        obj, w, c, all_c = train_op(design_variables,
                                    smoothing_matrix=smoothing_matrix,
                                    index_matrix=index_matrix,
                                    index_vector=index_vector,
                                    F=F,
                                    phis=phis,
                                    stretch_freedom=stretch_freedom,
                                    element_stiffness=element_stiffness,
                                    freedom_indexes=freedom_indexes,
                                    k_dim=k_dim,
                                    optimizer=optimizer,
                                    smoothing=smoothing)

        # if e % 20 == 0:
        #    smoothing = np.maximum(smoothing - smoothing * 0.5, 1)

        all_constraints.append(all_c)
        constraints.append(c)
        weights.append(w)
        designs.append(tf.sigmoid(design_variables).numpy())

        print(f"{e}: objective: {obj} weight: {w} constraint: {c} -- {time.time() - timestamp}")

    constraints, weights, designs, all_constraints = np.array(constraints), np.array(weights), \
                                                     np.array(designs), np.array(all_constraints)

    np.save("all_constraints", all_constraints)
    np.save("constraints", constraints)
    np.save("weights", weights)
    np.save("design", designs)
