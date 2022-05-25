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


def visualize_detection(img, boxes=None, masks=None, show_index=True):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img.squeeze(0).permute(1, 2, 0))
    if not masks is None:
        mask_rgba = colour_map_of_binary_masks(masks.squeeze(1))
        ax.imshow(mask_rgba)
    if not boxes is None:
        N = len(boxes)
        cmap = cm.get_cmap("viridis", N + 1)
        colour_array = cmap(np.linspace(0, 1, N + 1))
        for idx, (colour, box) in enumerate(zip(colour_array, boxes)):
            x, y, w, h = torchvision.ops.box_convert(box, "xyxy", "xywh")
            ax.add_patch(Rectangle((x, y), w, h, fill=False, color=colour))
            if show_index:
                ax.text(x + w / 2, y, f"{idx}", bbox=dict(fill=True, alpha=0.5))
    return fig, ax


def visualize_tracker(img, boxes=None, masks=None):
    N = 100
    cmap = cm.get_cmap("prism", N + 1)
    colour_array = cmap(np.linspace(0, 1, N + 1))
    np.random.permutation(colour_array)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img.squeeze(0).permute(1, 2, 0))

    import torchvision
    import torch
    from scipy.optimize import linear_sum_assignment

    mask_boxes = torchvision.ops.masks_to_boxes(masks)
    iou = torchvision.ops.box_iou(
        torch.stack(list(boxes.values()), dim=0), mask_boxes
    )
    cost = 1 - iou
    cost = np.where(cost > 0.25, 255, cost)
    track_idxs, mask_idxs = linear_sum_assignment(cost)
    track_id2mask_idx = {}

    for mask_idx, track_idx in zip(mask_idxs, track_idxs):
        track_id = list(boxes.keys())[track_idx]
        track_id2mask_idx[track_id] = mask_idx

    mask_cmap = np.zeros((1080, 1920, 4))
    if not masks is None:
        for obj_id in boxes.keys():
            mask_idx = track_id2mask_idx[obj_id]
            mask_cmap += (
                masks[mask_idx].float().unsqueeze(-1).repeat(1, 1, 4).numpy()
                * colour_array[obj_id]
            )
        mask_cmap[..., -1] = 0.3
        ax.imshow(mask_cmap)

    if not boxes is None:
        for idx, (obj_id, box) in enumerate(boxes.items()):
            x, y, w, h = torchvision.ops.box_convert(box, "xyxy", "xywh")
            ax.add_patch(
                Rectangle((x, y), w, h, fill=False, color=colour_array[obj_id])
            )
    return fig, ax
