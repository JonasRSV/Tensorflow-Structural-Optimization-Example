import tensorflow as tf
import time
import numpy as np
from struct_opt.environment_setup import initialize_env, get_element_index_matrix
from struct_opt.von_mises import VonMises
from struct_opt.circle import Circle
from struct_opt.fem import FEM
from struct_opt.barrier_objective import Barrier
from struct_opt.smoothing import NoSmoothing, GaussianSmoothing
from struct_opt.penalty import NoPenalty, EntropyPenalty, LinearEntropyPenalty, EntropyStructurePenalty
import sys


def train_op(design: tf.Variable,
             smoothing_function: tf.function,
             fem_function: tf.function,
             stress_function: tf.function,
             objective_function: tf.function,
             penalty_function: tf.function,
             optimizer: tf.keras.optimizers.Optimizer):
    weigth_norm: float = tf.constant(design.shape[0], dtype=tf.float32)

    with tf.GradientTape() as tape:
        sgm_design = tf.sigmoid(design)
        smooth_design = smoothing_function(sgm_design)
        weight = tf.reduce_sum(sgm_design) / weigth_norm
        U = fem_function(smooth_design)
        stress = stress_function(U)
        objective = objective_function(weight, stress)
        objective = objective + penalty_function(sgm_design)

        max_stress = tf.reduce_max(stress)

    gradients = tape.gradient(objective, design)
    optimizer.apply_gradients([(gradients, design)])

    return objective, smooth_design, weight, max_stress, stress


def __check_kwarg(name: str, kwargs):
    if name in kwargs:
        return kwargs[name]
    else:
        print(f"Please provide named argument {name} to main")
        sys.exit(0)


def main(problem_size: int,
         elements: np.ndarray,
         directions: np.ndarray,
         amplitudes: np.ndarray,
         max_constraint: float,
         mode: str,
         smoothing_mode: str = "none",
         penalty_mode: str = "none",
         thickness: float = 0.02,
         poisson_ratio: float = 0.3,
         initial_value_design: float = 2.0,
         elasticity_module: float = 1000,
         barrier_size: float = 100,
         barrier_width: float = 100,
         learning_rate: float = 0.1,
         data_directory: str = "../data",
         **kwargs):

    epochs = __check_kwarg("epochs", kwargs)

    design_variables, smoothing_matrix, node_index_matrix, node_index_vector, \
    F, stretch_freedom, element_stiffness, freedom_indexes, k_dim = \
        initialize_env(problem_size=problem_size,
                       thickness=thickness,
                       poisson_ratio=poisson_ratio,
                       elements=elements,
                       directions=directions,
                       amplitudes=amplitudes,
                       initial_values_design=initial_value_design)

    modes = {
        "von mises": VonMises,
        "circle": Circle
    }
    if mode not in modes:
        print(f"Mode must be one of ({' | '.join(modes.keys())})")
        sys.exit(0)

    stress_function = modes[mode](elasticity_module=elasticity_module,
                                  node_index_matrix=node_index_matrix, stretch_freedom=stretch_freedom,
                                  **kwargs).get_stress_function()

    smoothing_modes = {
        "none": NoSmoothing,
        "gaussian": GaussianSmoothing
    }

    if smoothing_mode not in smoothing_modes:
        print(f"Smoothing mode must be one of ({' | '.join(smoothing_modes.keys())})")
        sys.exit(0)

    element_index_matrix = get_element_index_matrix(problem_size=problem_size)
    smoothing_function = smoothing_modes[smoothing_mode](element_index_matrix, **kwargs).get_smoothing_function()

    penalty_modes = {
        "none": NoPenalty,
        "entropy": EntropyPenalty,
        "linear entropy": LinearEntropyPenalty,
        "entropy structure": EntropyStructurePenalty
    }

    if penalty_mode not in penalty_modes:
        if smoothing_mode not in smoothing_modes:
            print(f"Penalty mode must be one of ({' | '.join(penalty_modes.keys())})")
            sys.exit(0)

    penalty_function = penalty_modes[penalty_mode](**kwargs,
                                                   element_index_matrix=element_index_matrix
                                                   ).get_penalty_function()

    fem_function = FEM(
        node_index_vector=node_index_vector,
        F=F,
        element_stiffness=element_stiffness,
        freedom_indexes=freedom_indexes,
        k_dim=k_dim,
        elasticity_module=elasticity_module
    ).get_fem_function()

    objective_function = Barrier(
        max_constraint=max_constraint,
        barrier_width=barrier_width,
        barrier_size=barrier_size).get_objective_function()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    opt_timestamp = time.time()
    constraints, weights, designs, all_constraints = [], [], [design_variables.numpy()], []
    for e in range(epochs):
        timestamp = time.time()
        objective, smooth_design, weight, constraint, all_constraint = train_op(
            design_variables,
            smoothing_function=smoothing_function,
            fem_function=fem_function,
            stress_function=stress_function,
            objective_function=objective_function,
            penalty_function=penalty_function,
            optimizer=optimizer
        )

        all_constraints.append(all_constraint.numpy())
        constraints.append(constraint.numpy())
        weights.append(weight.numpy())
        designs.append(smooth_design.numpy())

        print(f"{e}: O: {objective} W: {weight} C {constraint} -- T: {time.time() - timestamp}")
    print(f"Time to run optimization {epochs} epochs: {time.time() - opt_timestamp} seconds")

    constraints, weights, designs, all_constraints = np.array(constraints), np.array(weights), \
                                                     np.array(designs), np.array(all_constraints)

    np.save(f"{data_directory}/all_constraints", all_constraints)
    np.save(f"{data_directory}/constraints", constraints)
    np.save(f"{data_directory}/weights", weights)
    np.save(f"{data_directory}/design", designs)
