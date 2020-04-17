import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

weights = np.load("weights.npy")
constraint = np.load("constraints.npy")
design = np.load("design.npy")
all_constraint = np.load("all_constraints.npy")


def reshape_design_vector(d: np.ndarray, width: np.ndarray, tiny_width: np.ndarray):

    image = np.zeros((width, width))

    index = 0
    for i in range(tiny_width):
        for j in range(width):
            image[j, i] = d[index]
            index += 1

    for i in range(tiny_width, width):
        for j in range(width - tiny_width, width):
            image[j, i] = d[index]
            index += 1

    return image


def _make_structure_animation(data, width, tiny_width):
    fig = plt.figure(figsize=(10, 10))
    images = []
    for d in data:
        image = reshape_design_vector(d, width, tiny_width)
        im = plt.imshow(image, animated=True)
        images.append([im])

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save("design_animation.mp4")

    plt.show()


if __name__ == "__main__":
    super_constant = 15

    width = super_constant * 5
    tiny_width = super_constant * 2

    _make_structure_animation(all_constraint, width, tiny_width)

