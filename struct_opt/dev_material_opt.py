from struct_opt import material_opt
import numpy as np


def _von_mises():
    problem_size = 3
    material_opt.main(
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
        max_constraint=10000,

        mode="von mises",

        smoothing_mode="gaussian",  # gaussian or none
        smoothing_width=3.0,
        variance=0.5,

        penalty_mode="entropy",  # entropy or none
        penalty_epochs=40,  # Last x epochs uses penalty
        penalty_size=2.0,

        thickness=0.02,
        poisson_ratio=0.3,
        initial_value_design=2.0,
        elasticity_module=1000,

        barrier_size=200,
        barrier_width=4000,

        epochs=100,
        learning_rate=0.15,

        data_directory="data"
    )


if __name__ == "__main__":
    _von_mises()
