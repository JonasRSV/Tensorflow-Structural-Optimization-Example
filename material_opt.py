import tensorflow as tf
import time
import numpy as np
from environment_setup import initialize_env
from von_mises import VonMises
from circle import Circle
from fem import FEM

@tf.function
def weight(design: tf.Tensor):
    return tf.reduce_sum(design)


@tf.function
def objective(design: tf.Variable,
              max_stress: tf.Tensor,
              smoothing_matrix: tf.Tensor,
              fem_function: tf.function,
              stress_function: tf.function,
              barrier_size: tf.Tensor):
    design = tf.sigmoid(design)

    # design = tf.linalg.matvec(smoothing_matrix, design)

    w = weight(design)
    u = fem_function(design)
    s = stress_function(u)

    w_obj = w
    c_obj = barrier_size * tf.reduce_mean(tf.abs(tf.math.log(tf.maximum((max_stress - s) / barrier_size, 0.00000001))))

    return w_obj + c_obj, w, tf.reduce_max(s), s


def train_op(design: tf.Variable,
             smoothing_matrix: tf.Tensor,
             fem_function: tf.function,
             stress_function: tf.function,
             optimizer: tf.keras.optimizers.Optimizer,
             barrier_size: float = 100):
    barrier_size = tf.constant(barrier_size, dtype=tf.float64)

    with tf.GradientTape() as tape:
        obj, w, c, all_c = objective(design,
                                     max_stress=3000,
                                     smoothing_matrix=smoothing_matrix,
                                     fem_function=fem_function,
                                     stress_function=stress_function,
                                     barrier_size=barrier_size)

    gradients = tape.gradient(obj, design)
    optimizer.apply_gradients([(gradients, design)])

    return obj, w, c, all_c


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
        print(f"Mode must be one of {' '.join(modes.keys())}")

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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    barrier_size = 1000

    constraints, weights, designs, all_constraints = [], [], [design_variables.numpy()], []
    for e in range(100):
        timestamp = time.time()
        obj, w, c, all_c = train_op(design_variables,
                                    smoothing_matrix=smoothing_matrix,
                                    fem_function=fem_function,
                                    stress_function=stress_function,
                                    optimizer=optimizer,
                                    barrier_size=barrier_size)

        # if e % 20 == 0:
        #    barrier_size = np.maximum(barrier_size - barrier_size * 0.5, 1)

        all_constraints.append(all_c)
        constraints.append(c)
        weights.append(w)
        designs.append(tf.sigmoid(design_variables).numpy())

        print(f"{e}: objective: {obj} weight: {w} max constraint: {c} -- {time.time() - timestamp}")

    constraints, weights, designs, all_constraints = np.array(constraints), np.array(weights), \
                                                     np.array(designs), np.array(all_constraints)

    np.save("all_constraints", all_constraints)
    np.save("constraints", constraints)
    np.save("weights", weights)
    np.save("design", designs)


if __name__ == "__main__":
    problem_size = 3
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
        mode="von mises"
    )
