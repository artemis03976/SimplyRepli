import matplotlib.pyplot as plt
import math
import numpy as np
from torch import Tensor


def show_img(images, cols=4):
    # transform image to numpy array
    if isinstance(images, Tensor):
        images = images.cpu().detach().numpy()

    # get shape
    batch_size, channel, height, weight = images.shape

    rows = int(math.ceil(batch_size / cols))
    # create subplot frame
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    # protection for single image
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    # adjust image shape for plot func
    images = np.transpose(images, [0, 2, 3, 1])

    # plot every generated image
    for image, axis in zip(images, axes):
        # rescale from [0, 1] to [0, 255]
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        axis.imshow(image)
        # eliminate axis
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.show()
