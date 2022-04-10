import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from src.utils.torch_utils import dict2keys_and_items


def convert_frames(frames, inp_fmt, out_fmt):
    if inp_fmt == "tracker" and out_fmt == "detector":
        images = []
        targets = []
        for frame in frames:
            image = frame["img"]
            ids, boxes = dict2keys_and_items(frame["gt"], "torch")
            ids, vis = dict2keys_and_items(frame["vis"], "torch")
            target = {"boxes": boxes, "visibilities": vis}
            images.append(image)
            targets.append(target)
        return images, targets


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


def obj_detect_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def run_obj_detect(model, data_loader, debug=False):
    model.eval()
    device = list(model.parameters())[0].device
    results_dict = {}
    for images, targets in tqdm(data_loader, desc="eval_batches", leave=False):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            preds = model(images)

        for pred, target in zip(preds, targets):
            image_id = target["image_id"].item()
            results_dict[image_id] = {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
            }
        if debug:
            break
    return results_dict


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


def mask_convert(masks, inp_fmt, out_fmt):
    """
    Attention!!!! channels are the first dimension !!!

    masks [C, W, H]  
    """
    if inp_fmt == "binary" and out_fmt == "scalar":
        # regular conversion binary -> scalar
        # the ~overlap matrix sets overlapping pixels to 0 to prevent summing of class ids
        # typically this just affects the boundaries between objects
        # exception: if one mask is fully nested inside another, maskrcnn does not do a whole in the mask of the larger obj
        # -> the  ~overlap multiplication will set the nested object to zero
        # -> exception is handled below
        masks_binary = masks
        num_classes = len(masks_binary)
        overlap = masks_binary.sum(dim=0) > 1
        classes = torch.arange(1, num_classes + 1)

        masks_classes = (classes * masks_binary.permute(1, 2, 0)).permute(
            2, 0, 1
        )
        masks_scalar = masks_classes.sum(dim=0) * ~overlap

        # handle exception
        # also make sure that the nested masks dont overlap with each other
        mask_or_overlap = torch.logical_or(
            masks_binary, overlap.repeat(num_classes, 1, 1)
        )
        nested_mask_idx = torch.where(
            (mask_or_overlap == overlap.repeat(num_classes, 1, 1))
            .reshape(num_classes, -1)
            .all(-1)
        )[0]
        nested_overlap = masks_binary[nested_mask_idx].sum(dim=0) > 1
        for idx in nested_mask_idx:
            masks_scalar += masks_classes[idx] * ~nested_overlap
        return masks_scalar.int()

    if inp_fmt == "scalar" and out_fmt == "binary":
        masks_scalar = masks
        classes = torch.unique(masks_scalar)[1:]
        num_classes = len(classes)
        masks_binary = (
            masks_scalar.repeat(num_classes, 1, 1).permute(1, 2, 0) == classes
        )
        masks_binary = masks_binary.permute(2, 0, 1)
        return masks_binary.int()


def denormalize_boxes(anchor_boxes, normalized_boxes):
    """
    Arguments
    ---------
    anchor_boxes [N, 4]:        a batch of anchor boxes, where one anchor applies to L boxes
    normalized_boxes [N, L, 4]: a batch of normalized boxes, L boxes grouped together share an anchor

    Returns
    -------
    boxes [N, L, 4]
    """
    x_a, y_a, w_a, h_a = anchor_boxes.T
    t_x, t_y, t_w, t_h = normalized_boxes.permute(2, 1, 0)

    x = x_a + w_a * t_x
    y = y_a + h_a * t_y
    w = w_a * torch.exp(t_w)
    h = h_a * torch.exp(t_h)

    boxes = torch.stack([x, y, w, h], axis=-1).permute(1, 0, 2)
    return boxes


def normalize_boxes(anchor_boxes, boxes):
    """
    Arguments
    ---------
    anchor_boxes [N, 4]:        a batch of anchor boxes, where one anchor applies to L boxes
    boxes [N, L, 4]: a batch of normalized boxes, L boxes grouped together share an anchor

    Returns
    -------
    normalized_boxes [N, L, 4]
    """
    x_a, y_a, w_a, h_a = anchor_boxes.T
    x, y, w, h = boxes.permute(2, 1, 0)

    t_x = (x - x_a) / w_a
    t_y = (y - y_a) / h_a
    t_w = torch.log(w / w_a)
    t_h = torch.log(h / h_a)

    boxes = torch.stack([t_x, t_y, t_w, t_h], axis=-1).permute(1, 0, 2)
    return boxes
