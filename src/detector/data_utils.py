import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import csv
from collections import defaultdict
from src.detector.utils import mask_convert


def encode_segmentation(seg_img):
    """
    Arguments
    ---------
    seg_img
        - Decoded mask with the following semantics
            - 0 = background & non pedestrian objects
            - 1, 2, ... = pedestrian

    Returns
    -------
    seg_img
        - Encoded mask with the following semantic scalars
            - 0 = background
            - 2001, 2002, ... = pedestrian (with ids = 001, 002, ...)
            - 10000 =  pedestrians with low class prob / low score / low visibility
    """
    seg_img[seg_img != 0] += 2000
    return seg_img


def decode_segmentation(seg_img):
    """
    Arguments
    ---------
    seg_img
        - Encoded mask with the following semantic scalars
            - 0 = background
            - 2001, 2002, ... = pedestrian (with ids = 001, 002, ...)
            - 10000 = pedestrians with low class prob / low score / low visibility
    Returns
    -------
    seg_img
        - Decoded mask with the following semantics
            - 0 = background & non pedestrian objects
            - 1, 2, ... = pedestrian
    """
    # filter only pedestrians
    class_img = np.floor_divide(seg_img, 1000)
    seg_img[class_img != 2] = 0
    # get instance masks
    seg_img = np.mod(seg_img, 1000)
    return seg_img


def load_segmentation(seg_path, box_ids=None, only_obj_w_mask=True):
    encoded_masks = TF.to_tensor(Image.open(seg_path))
    masks = decode_segmentation(encoded_masks)
    seg_ids = torch.unique(masks)[1:]

    if not box_ids is None:
        keep_ids = torch.tensor(list(set(box_ids) & set(seg_ids)))
        remove_seg_ids = torch.tensor(list(set(seg_ids) - set(keep_ids)))
        masks[torch.isin(masks, remove_seg_ids)] = 0
    else:
        keep_ids = seg_ids

    binary_masks = mask_convert(masks, "scalar", "binary")

    if only_obj_w_mask or box_ids is None:
        return_masks = binary_masks
    else:
        padded_masks = torch.zeros_like(masks, dtype=torch.float32).repeat(
            len(box_ids), 1, 1
        )
        for binary_mask, seg_id in zip(binary_masks, seg_ids):
            idx_of_id = torch.where(box_ids == seg_id)[0]
            padded_masks[idx_of_id] = binary_mask
        return_masks = padded_masks
    return return_masks, keep_ids


def load_detection_from_txt(txt_path, vis_threshold=0.0, mode="gt"):
    """
    modes: "det", "gt", "track"
    """
    boxes = defaultdict(dict)
    visibility = defaultdict(dict)
    gt = {}
    with open(txt_path, "r") as inf:
        reader = csv.reader(inf, delimiter=",")
        for row in reader:
            # class person, certainity 1, visibility >= 0.25
            if mode == "gt" and not (
                int(row[6]) == 1
                and int(row[7]) == 1
                and float(row[8]) >= vis_threshold
            ):
                continue
            elif mode == "gt":
                visibility[int(row[0])][int(row[1])] = float(row[8])

            if mode == "det" and False:  # TODO add condition
                continue
            if mode == "track" and False:  # TODO add condition
                continue
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = float(row[2]) - 1
            y1 = float(row[3]) - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + float(row[4]) - 1
            y2 = y1 + float(row[5]) - 1
            bb = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).clip(min=0)
            boxes[int(row[0])][int(row[1])] = bb

    gt["boxes"] = boxes
    if mode == "gt":
        gt["visibilities"] = visibility
    return gt
