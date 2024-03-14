import matplotlib.pyplot as plt
import math
import numpy as np
from torch import Tensor


def show_img(images, cols=4):
    if isinstance(images, Tensor):
        images = images.cpu().detach().numpy()

    batch_size, channel, height, weight = images.shape

    rows = int(math.ceil(batch_size / cols))
    # create subplot frame
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    axes = axes.flatten()
    # adjust image
    images = np.transpose(images, [0, 2, 3, 1])

    # plot every generated image
    for image, axis in zip(images, axes):
        image = (image * 255).astype(np.uint8)
        axis.imshow(image)
        # eliminate axis
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.show()
