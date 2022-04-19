import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from src.utils.torch_utils import dict2keys_and_items


def full_image_box(image):
    """
    return xyxy box of the whole image
    """
    return torch.Tensor([0, 0, image.shape[2], image.shape[1]])


def convert_frames(frames, inp_fmt, out_fmt):
    if inp_fmt == "tracker" and out_fmt == "detector":
        images = []
        targets = []
        for frame in frames:
            image = frame["img"]
            ids, boxes = dict2keys_and_items(frame["gt"], "torch")
            ids, vis = dict2keys_and_items(frame["vis"], "torch")
            target = {"boxes": boxes, "visibilities": vis}
            if "seg_img" in frame.keys():
                target.update({"seg_img": frame["seg_img"]})
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


def mask_convert(masks, inp_fmt, out_fmt):
    """

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
        overlap = (masks_binary.sum(dim=0) > 1).bool()
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
        nested_overlap = (masks_binary[nested_mask_idx].sum(dim=0) > 1).bool()
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


def binary_mask_iou(masks1, masks2, speed_up_mask=None):
    """
  Arguments
  ---------
    masks1 : [N1, H, W]
    masks2 : [N2, H, W]
    box_iou : [N1, N2] can be passed for speedup

  Returns
  -------
    iou_matrix : [N1, N2]
  """
    masks1_area = torch.count_nonzero(masks1, axis=(-2, -1))
    masks2_area = torch.count_nonzero(masks2, axis=(-2, -1))
    (masks1_area_matrix, masks2_area_matrix) = torch.meshgrid(
        masks2_area, masks1_area
    )

    masks1_area_matrix = speed_up_mask * masks1_area_matrix
    masks2_area_matrix = speed_up_mask * masks2_area_matrix

    intersection_matrix = []
    for row, mask1 in enumerate(masks1):
        if not speed_up_mask is None:
            intersection_row = torch.zeros((len(masks2)))
            notnull = torch.where(speed_up_mask[row] > 0)[0]

            mask1 = torch.tile(mask1, (len(notnull), 1, 1))
            notnull_intersection_row = torch.count_nonzero(
                torch.logical_and(mask1, masks2[notnull]), axis=(-2, -1)
            ).float()
            intersection_row[notnull] = notnull_intersection_row
        else:
            mask1 = torch.tile(mask1, (len(masks2), 1, 1))
            intersection_row = torch.count_nonzero(
                torch.logical_and(mask1, masks2), axis=(-2, -1)
            )
        intersection_matrix.append(intersection_row)
    intersection_matrix = torch.stack(intersection_matrix, axis=0)
    union_matrix = masks1_area_matrix + masks2_area_matrix - intersection_matrix
    iou_matrix = intersection_matrix / union_matrix.clip(min=0.00001)
    return iou_matrix


def hybrid_iou(
    boxes1, boxes2, masks1, masks2, area_threshold=5000, iou_threshold=0.5
):
    """
    1. Compute box_iou for all box pairs
    2. For all box_iou values above iou_threshold, replace them with mask_iou
    """
    box_iou = torchvision.ops.box_iou(boxes1, boxes2)
    speed_up_mask = box_iou
    speed_up_mask = torch.logical_and(
        speed_up_mask > iou_threshold, speed_up_mask < 1
    ).bool()
    areas1 = torchvision.ops.box_area(boxes1)
    areas2 = torchvision.ops.box_area(boxes2)
    speed_up_mask[areas1 < area_threshold, :] = False
    speed_up_mask[:, areas2 < area_threshold] = False
    if (boxes1 == boxes2).all():
        mask_iou_triu = binary_mask_iou(
            masks1, masks2, torch.triu(speed_up_mask, 1)
        )
        mask_iou = mask_iou_triu + mask_iou_triu.T
    else:
        mask_iou = binary_mask_iou(masks1, masks2, speed_up_mask)
    return speed_up_mask * mask_iou + ~speed_up_mask * box_iou


def nms(iou, iou_threshold):
    keep_idxs = []
    for idx, row in enumerate(iou):
        if idx == 0 or (row[keep_idxs] < iou_threshold).all():
            keep_idxs.append(idx)
    return torch.tensor(keep_idxs)


def mask_nms(boxes, scores, labels, masks, iou_threshold=0.5):
    """
    This function performs the nms algorithm with hybrid iou as a distance metric
    """
    iou = hybrid_iou(boxes, boxes, masks, masks, iou_threshold=0.5)

    keep_idxs = []
    for class_id in torch.unique(labels):
        class_idxs = torch.where(class_id == labels)[0]
        class_iou = iou[class_idxs][:, class_idxs].clone()
        keep_class_local_idxs = nms(class_iou, iou_threshold=0.5)
        keep_class_idxs = class_idxs[keep_class_local_idxs]
        keep_idxs.append(keep_class_idxs)
    keep_idxs = torch.cat(keep_idxs)
    keep = keep_idxs[scores[keep_idxs].argsort(descending=True)]
    return keep

