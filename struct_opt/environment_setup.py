import tensorflow as tf
import numpy as np
import time


def make_smoothing(n_design: int):
    return np.random.rand(n_design, n_design) / n_design


def make_F(bracket: int,
           tiny_bracket: int,
           node_index_matrix: np.ndarray,
           elements: np.ndarray,
           directions: np.ndarray,
           amplitudes: np.ndarray):
    force_size = 2 * (tiny_bracket + 1) * (2 * bracket - tiny_bracket + 1)

    F = np.zeros((force_size, amplitudes.size))
    for i in range(elements.size):
        row = node_index_matrix[elements[i]]

        if directions[i] == 0:
            positions = row[0::2]
        else:
            positions = row[1::2]

        for position in positions:
            F[position, i] = amplitudes[i] / len(positions)

    return F


def get_index_vector(node_index_matrix: np.ndarray):
    column_index_for_K = np.kron(node_index_matrix.T, np.ones(8)).astype(np.int).T.reshape(-1, 1)
    row_index_for_K = np.kron(node_index_matrix, np.ones(8)).astype(np.int).reshape(-1, 1)

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

    def make_node_index_matrix(heigth: int, width: int):
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

    node_index_matrix_top = make_node_index_matrix(bracket_heigth, tiny_bracket_width)
    node_index_matrix_bottom = make_node_index_matrix(remaining_height, remaining_width)

    node_index_matrix_bottom = node_index_matrix_bottom + 2 * (tiny_bracket_width + 1) * bracket_heigth
    node_index_matrix = np.concatenate([node_index_matrix_top, node_index_matrix_bottom], axis=0)

    return node_index_matrix - 1, stretch_freedom.T, element_stiffness


def make_fixed_degrees_of_freedom(bracket: int, tiny_bracket: int, k_dim: int):
    tiny_indexes = np.arange(0, tiny_bracket + 1)
    tiny_indexes = tiny_indexes * 2 * (bracket + 1)

    locked_indexes = tiny_indexes + 1

    fixed_degrees_of_freedom = np.concatenate([tiny_indexes, locked_indexes])
    all_values = np.arange(0, 2 * (tiny_bracket + 1) * (2 * bracket - tiny_bracket + 1))

    freedom_indexes = np.array([index for index in all_values if index not in fixed_degrees_of_freedom])

    return freedom_indexes


def get_element_index_matrix(problem_size: int):
    bracket = problem_size * 5
    tiny_bracket = problem_size * 2

    element_index_matrix = np.ones((bracket, bracket), dtype=np.int) * -1

    index = 0
    for i in range(tiny_bracket):
        for j in range(bracket):
            element_index_matrix[j, i] = index
            index += 1

    for i in range(tiny_bracket, bracket):
        for j in range(bracket - tiny_bracket, bracket):
            element_index_matrix[j, i] = index
            index += 1

    return element_index_matrix


def initialize_env(problem_size: int,
                   thickness: float,
                   poisson_ratio: float,
                   elements: np.ndarray,
                   directions: np.ndarray,
                   amplitudes: np.ndarray,
                   initial_values_design=1.0):
    timestamp = time.time()

    bracket = problem_size * 5
    tiny_bracket = problem_size * 2
    element_length = 1 / bracket
    n_design = np.square(problem_size) * 16

    """Vector to optimise"""
    design_variables = tf.Variable(np.ones(n_design) * initial_values_design, trainable=True, dtype=tf.float32)

    """Smoothing matrix for the design variables"""
    smoothing_matrix = make_smoothing(n_design)

    """Constant matrices for pre and post processing in the solver"""
    node_index_matrix, stretch_freedom, element_stiffness = setup_strain_problem(
        bracket_heigth=bracket,
        tiny_bracket_width=tiny_bracket,
        element_length=element_length,
        thickness=thickness,
        poisson_ratio=poisson_ratio
    )

    """Vector for constructing K matrix"""
    node_index_vector = get_index_vector(node_index_matrix=node_index_matrix)

    """Force vector for FEM"""
    F = make_F(bracket=bracket,
               tiny_bracket=tiny_bracket,
               node_index_matrix=node_index_matrix,
               elements=elements,
               directions=directions,
               amplitudes=amplitudes)

    """Dimension of K matrix"""
    k_dim = node_index_matrix.max() + 1

    """Fixed degrees of freedom mask"""
    freedom_indexes = make_fixed_degrees_of_freedom(bracket=bracket,
                                                    tiny_bracket=tiny_bracket,
                                                    k_dim=k_dim)

    print(f"Initializing env: {time.time() - timestamp} seconds")

    return design_variables, smoothing_matrix, node_index_matrix, \
           node_index_vector, F, stretch_freedom, element_stiffness, \
           freedom_indexes, k_dim
