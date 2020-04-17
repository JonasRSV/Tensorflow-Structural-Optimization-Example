import tensorflow as tf
import time
import numpy as np
from environment_setup import initialize_env
from von_mises import VonMises
from circle import Circle
from fem import FEM
from barrier_objective import Barrier
import sys


def train_op(design: tf.Variable,
             smoothing_matrix: tf.Tensor,
             fem_function: tf.function,
             stress_function: tf.function,
             objective_function: tf.function,
             optimizer: tf.keras.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        sgm_design = tf.sigmoid(design)
        # design = tf.linalg.matvec(smoothing_matrix, design)
        weight = tf.reduce_sum(sgm_design)
        U = fem_function(sgm_design)
        stress = stress_function(U)
        objective = objective_function(weight, stress)

        max_stress = tf.reduce_max(stress)

    gradients = tape.gradient(objective, design)
    optimizer.apply_gradients([(gradients, design)])

    return objective, weight, max_stress, stress


def main(problem_size: int,
         elements: np.ndarray,
         directions: np.ndarray,
         amplitudes: np.ndarray,
         max_constraint: float,
         mode: str,
         thickness: float = 0.02,
         poisson_ratio: float = 0.3,
         radius: float = 1.0,
         initial_value_design: float = 2.0,
         elasticity_module: float = 1000,
         barrier_size: float = 100,
         barrier_width: float = 100,
         epochs: int = 100,
         learning_rate: float=0.1,
         **kwargs):
    design_variables, smoothing_matrix, index_matrix, index_vector, \
    F, stretch_freedom, element_stiffness, freedom_indexes, k_dim = \
        initialize_env(problem_size=problem_size,
                       thickness=thickness,
                       poisson_ratio=poisson_ratio,
                       radius=radius,
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
                                  index_matrix=index_matrix, stretch_freedom=stretch_freedom,
                                  **kwargs).get_stress_function()

    fem_function = FEM(
        index_vector=index_vector,
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
        objective, weight, constraint, all_constraint = train_op(
            design_variables,
            smoothing_matrix=smoothing_matrix,
            fem_function=fem_function,
            stress_function=stress_function,
            objective_function=objective_function,
            optimizer=optimizer,
        )

        all_constraints.append(all_constraint)
        constraints.append(constraint)
        weights.append(weight)
        designs.append(tf.sigmoid(design_variables).numpy())

        print(f"{e}: O: {objective} W: {weight} C {constraint} -- T: {time.time() - timestamp}")
    print(f"Time to run optimization {epochs} epochs: {time.time() - opt_timestamp} seconds")

    constraints, weights, designs, all_constraints = np.array(constraints), np.array(weights), \
                                                     np.array(designs), np.array(all_constraints)

    np.save("all_constraints", all_constraints)
    np.save("constraints", constraints)
    np.save("weights", weights)
    np.save("design", designs)


def _von_mises():
    problem_size = 5
    main(
        problem_size=problem_size,
        elements=np.array([
            16 * np.square(problem_size) - problem_size * 2
        ]),
        directions=np.array([
            1
        ]),
        amplitudes=np.array([
            -1
        ]),
        max_constraint=3000,
        mode="von mises",
        barrier_size=100,
        barrier_width=300,
    )


def _circle():
    problem_size = 8
    main(
        problem_size=problem_size,
        elements=np.array([
            16 * np.square(problem_size) - problem_size * 2
        ]),
        directions=np.array([
            1
        ]),
        amplitudes=np.array([
            -1
        ]),
        max_constraint=3000,
        mode="circle",
        barrier_size=100,
        barrier_width=200,
        phis=[0.0],
        kf=0.3
    )


if __name__ == "__main__":
    _von_mises()
