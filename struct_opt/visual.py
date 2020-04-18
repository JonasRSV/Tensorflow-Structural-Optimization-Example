from struct_opt.environment_setup import get_element_index_matrix
from struct_opt.smoothing import GaussianSmoothing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def reshape_design_vector(d: np.ndarray, element_index_matrix):
    image = np.zeros_like(element_index_matrix, dtype=np.float64)

    height, width = element_index_matrix.shape

    for i in range(height):
        for j in range(width):
            if element_index_matrix[i, j] != -1:
                image[i][j] = d[element_index_matrix[i, j]]

    return image


def _make_structure_animation(data, problem_size: int, data_path: str, plot=True):
    element_index_matrix = get_element_index_matrix(problem_size)

    fig = plt.figure(figsize=(10, 10))
    images = []
    for d in data:
        image = reshape_design_vector(d, element_index_matrix)
        im = plt.imshow(image, animated=True)
        images.append([im])

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                                    repeat_delay=10000)

    ani.save(f"{data_path}/design_animation.mp4")

    if plot:
        plt.show()

    return ani


def highlight_elements(element_index_matrix: np.ndarray, elements: [int]):
    highlight_color = np.array([1.0, 0.0, 0.0, 0.8])  # This is RGBA colors
    background_color = np.array([0.0, 0.0, 0.0, 0.5])  # This is RGBA colors
    element_default = np.array([0.0, 0.3, 0.0, 0.5])  # This is RGBA colors

    heigth, width = element_index_matrix.shape

    image = np.zeros((heigth, width, 4))
    for i in range(heigth):
        for j in range(width):
            if element_index_matrix[i, j] == -1:
                image[i, j] = background_color
            elif element_index_matrix[i, j] in elements:
                image[i, j] = highlight_color
            else:
                image[i, j] = element_default

    plt.figure(figsize=(14, 14))
    plt.imshow(image)
    plt.show()


def highlight_gaussian_smoothing(element_index_matrix: np.ndarray, element: int, variance: float, width: float):
    smoothing = GaussianSmoothing(element_index_matrix, smoothing_width=width, variance=variance)
    smoothing = smoothing.smoothing_matrix.numpy()

    background_color = np.array([0.0, 0.0, 0.0, 0.5])  # This is RGBA colors
    element_default = np.array([0.0, 0.3, 0.0, 0.5])  # This is RGBA colors
    smoothing_color = np.array([1.0, 0.0, 0.0, 0.0])

    smoothing_element = smoothing[element]

    max_smooth = np.max(smoothing_element)
    rescaling = 1 / max_smooth

    heigth, width = element_index_matrix.shape
    image = np.zeros((heigth, width, 4))
    for i in range(heigth):
        for j in range(width):
            if element_index_matrix[i, j] != -1:
                #print(smoothing_element[element_index_matrix[i, j]])
                #print("sum smoothing", sum(smoothing_element))
                #print("max", np.max(smoothing_element))
                image[i, j] = rescaling * smoothing_color * smoothing_element[element_index_matrix[i, j]] + element_default
            else:
                image[i, j] = background_color

    plt.figure(figsize=(14, 14))
    plt.imshow(image)
    plt.show()


class Visual:

    def __init__(self, data_path="../data", plot=True):
        self.data_path = data_path
        self.design = np.load(f"{data_path}/design.npy")
        self.weights = np.load(f"{data_path}/weights.npy")
        self.constraints = np.load(f"{data_path}/constraints.npy")
        self.all_constraint = np.load(f"{data_path}/all_constraints.npy")
        self.plot = plot

    def animate(self, problem_size: int, mode: str):
        animations = {
            "design": lambda: _make_structure_animation(self.design, problem_size,
                                                        data_path=self.data_path, plot=self.plot),
            "stress": lambda: _make_structure_animation(self.all_constraint, problem_size,
                                                        data_path=self.data_path, plot=self.plot),
        }

        return animations[mode]()

    def constraint_weigth_plot(self):
        epochs = np.arange(0, self.weights.size)

        fig, ax = plt.subplots(figsize=(20, 10))
        weight_line = ax.plot(epochs, self.weights, color="green")[0]
        plt.xticks(fontsize=16)
        plt.ylabel("Weight", fontsize=16)
        plt.yticks(fontsize=16)

        twinx = ax.twinx()
        constraint_line = twinx.plot(epochs, self.constraints, color="red")[0]
        plt.ylabel("Constraint", fontsize=16)
        plt.legend([weight_line, constraint_line], ["weight", "constraint"], fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_xlabel("Epochs", fontsize=16)
        plt.show()


if __name__ == "__main__":
    problem_size = 5
    V = Visual(data_path="../data")
    V.animate(problem_size, mode="design")
    # V.constraint_weigth_plot()
