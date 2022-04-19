import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

from src.detector.utils import mask_convert


def colour_map_of_binary_masks(masks):
    """
    masks [C, H, W]
    colour_map [H, W, 4] (RGBA)
    """
    N = len(masks)
    cmap = cm.get_cmap("viridis", N + 1)
    colour_array = cmap(np.linspace(0, 1, N + 1))
    colour_array[:, -1] = 0.5
    colour_array[0, -1] = 0.0

    scalar_masks = mask_convert(masks, "binary", "scalar")
    colour_map = colour_array[scalar_masks.flatten()].reshape(
        *scalar_masks.shape, 4
    )
    return colour_map


def visualize_detection(img, det):
    """Visualize faster rcnn format detection
    Arguments
    ----------
    img: [3, 1080, 1920]
    det: {"masks": [N, 1080, 1920], "scores": [N], "boxes": [N, 4]}

    Returns
    -------
    fig

    """

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img.squeeze(0).permute(1, 2, 0))

    if "masks" in det.keys():
        mask_rgba = colour_map_of_binary_masks(det["masks"].squeeze(1))

        ax.imshow(mask_rgba)
    for box in det["boxes"]:
        x, y, w, h = torchvision.ops.box_convert(box, "xyxy", "xywh")
        ax.add_patch(Rectangle((x, y), w, h, fill=False))
    return fig
