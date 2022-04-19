import cv2
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


def compute_optical_flows_of_sequence(first_img_path):
    """
    first_img_path : path to first image of sequence
    """
    flows = []
    video = cv2.VideoCapture(first_img_path, cv2.CAP_IMAGES)
    previous_bgr_frame = video.read()[1]
    previous_grey_frame = cv2.cvtColor(previous_bgr_frame, cv2.COLOR_BGR2GRAY)

    while True:
        keep_going, current_bgr_frame = video.read()
        if not keep_going:
            break
        current_grey_frame = cv2.cvtColor(current_bgr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_grey_frame,
            current_grey_frame,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        previous_grey_frame = current_grey_frame
        flows.append(flow)
    return flows


def compute_optical_flow(frame_1_rgb, frame_2_rgb):
    """
    Arguments
    ----------
    frame_1_rgb : [3, H, W] torch or numpy
    frame_2_rgb : [3, H, W] torch or numpy

    Returns
    -------
    flow [H, W, 2] torch or numpy
    """
    return_torch = False
    if isinstance(frame_1_rgb, torch.Tensor):
        frame_1_rgb = frame_1_rgb.numpy().copy()
        frame_2_rgb = frame_2_rgb.numpy().copy()
        return_torch = True

    if frame_1_rgb.shape[0] == 3:
        frame_1_rgb = frame_1_rgb.transpose(1, 2, 0).copy()
        frame_2_rgb = frame_2_rgb.transpose(1, 2, 0).copy()

    frame_1_grey = cv2.cvtColor(frame_1_rgb, cv2.COLOR_RGB2GRAY)
    frame_2_grey = cv2.cvtColor(frame_2_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        frame_1_grey, frame_2_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    if return_torch:
        frame_1_grey = torch.from_numpy(frame_1_grey.copy())
        frame_2_grey = torch.from_numpy(frame_2_grey.copy())
    return flow


def flow2rgb(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    brightness = 255
    height, width, _ = flow.shape
    hsv_flow = np.zeros((height, width, 3))
    hsv_flow[..., 0] = np.rad2deg(angle) / 2
    hsv_flow[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv_flow[..., 1] = brightness
    rgb_flow = cv2.cvtColor(hsv_flow.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb_flow


def grad_to_rgb(angle, absolute):
    max_abs = 255
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return hsv_to_rgb((angle / 2 / np.pi, absolute / max_abs, 1))


def visualize_flow(flow):
    rgb_flow = flow2rgb(flow)
    fig, ax = plt.subplots(
        1,
        2,
        squeeze=False,
        figsize=(15, 15),
        gridspec_kw={"width_ratios": [5, 1]},
    )
    ax[0, 0].imshow(rgb_flow)

    wheel_ax = fig.add_subplot(1, 2, 2, projection="polar")
    max_abs = 255
    n = 200
    t = np.linspace(0, 2 * np.pi, n)
    r = np.linspace(0, max_abs, n)
    rg, tg = np.meshgrid(r, t)
    c = np.array(list(map(grad_to_rgb, tg.T.flatten(), rg.T.flatten())))
    cv = c.reshape((n, n, 3))
    wheel_ax.grid(False)
    wheel_ax.pcolormesh(t, r, cv[:, :, 1], color=c)
    wheel_ax.axis("off")

    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 1].remove()
    fig.tight_layout()
    return fig, ax
