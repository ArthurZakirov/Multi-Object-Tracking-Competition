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

keypoint_name2idx = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

keypoint_idx2name = {idx: name for (name, idx) in keypoint_name2idx.items()}


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
        masks_scalar = torch.zeros(*masks.shape[-2:])
        covered_area = torch.zeros(*masks.shape[-2:])
        masks_binary = masks.clone().bool()
        num_classes = len(masks_binary)

        for label, mask in enumerate(masks_binary, start=1):
            masks_scalar += label * torch.logical_and(mask, ~(covered_area > 0))
            covered_area += mask

        # overlap = (masks_binary.sum(dim=0) > 1).bool()
        # classes = torch.arange(1, num_classes + 1)

        # masks_classes = (classes * masks_binary.permute(1, 2, 0)).permute(
        #     2, 0, 1
        # )
        # masks_scalar = masks_classes.sum(dim=0) * ~overlap

        # # handle exception
        # # also make sure that the nested masks dont overlap with each other

        # mask_or_overlap = torch.logical_or(
        #     masks_binary, overlap.repeat(num_classes, 1, 1)
        # )

        # nested_mask_idx = torch.where(
        #     (mask_or_overlap == overlap.repeat(num_classes, 1, 1))
        #     .reshape(num_classes, -1)
        #     .all(-1)
        # )[0]
        # nested_overlap = (masks_binary[nested_mask_idx].sum(dim=0) > 1).bool()
        # for idx in nested_mask_idx:
        #     masks_scalar += masks_classes[idx] * ~nested_overlap
        return masks_scalar.int()

    if inp_fmt == "scalar" and out_fmt == "binary":
        masks_scalar = masks.clone().int()
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


def intersection_boxes(boxes1, boxes2):
    """
    return the xyxy coordinates of the intersections of box pairs


    Arguments
    ----------
    boxes1 [N, 4] xyxy
    boxes2 [M, 4] xyxy

    Returns
    -------
    intersection_boxes [4, N, M] xyxy
    """
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    overlap_pairs = (rb - lt > 0).all(dim=-1)
    intersection_boxes = overlap_pairs * torch.cat([lt, rb], dim=-1).permute(
        2, 0, 1
    )
    return intersection_boxes.int()


def binary_mask_iou(boxes1, boxes2, masks1, masks2, speed_up_matrix=None):
    intersection_matrix, union_matrix = _binary_mask_inter_union(
        boxes1=boxes1,
        boxes2=boxes2,
        masks1=masks1,
        masks2=masks2,
        speed_up_matrix=speed_up_matrix,
    )
    iou_matrix = intersection_matrix / union_matrix
    return iou_matrix


def _binary_mask_inter_union(
    boxes1, boxes2, masks1, masks2, speed_up_matrix=None
):
    """
    Arguments
    ---------
        masks1 : [N1, H, W]
        masks2 : [N2, H, W]
        intersection_boxes : [4, N1, N2] can be passed for speedup

    Returns
    -------
        iou_matrix : [N1, N2]
    """
    assert len(masks1.shape) == 3 and len(masks2.shape) == 3

    masks1_area = torch.count_nonzero(masks1, axis=(-2, -1))
    masks2_area = torch.count_nonzero(masks2, axis=(-2, -1))
    (masks1_area_matrix, masks2_area_matrix) = torch.meshgrid(
        masks1_area, masks2_area
    )
    if speed_up_matrix is None:
        speed_up_matrix = torchvision.ops.box_iou(boxes1, boxes2)

    inter_boxes = intersection_boxes(boxes1, boxes2)
    intersection_matrix = []

    for row, mask1 in enumerate(masks1):
        intersection_row = torch.zeros((len(masks2)))
        notnull = torch.where(speed_up_matrix[row] > 0)[0]
        x_min, y_min, x_max, y_max = inter_boxes[:, row, notnull]
        masks2_ = masks2[notnull]

        mask1_ovl_crops = [
            mask1[y_min[i] : y_max[i], x_min[i] : x_max[i]]
            for i in range(len(notnull))
        ]
        masks2_ovl_crops = [
            masks2_[i, y_min[i] : y_max[i], x_min[i] : x_max[i]]
            for i in range(len(notnull))
        ]
        intersection_row[notnull] = torch.tensor(
            [
                torch.count_nonzero(crop1 * crop2)
                for (crop1, crop2) in zip(mask1_ovl_crops, masks2_ovl_crops)
            ]
        ).float()

        intersection_matrix.append(intersection_row)

    intersection_matrix = torch.stack(intersection_matrix, axis=0)
    union_matrix = (
        masks1_area_matrix
        + masks2_area_matrix
        - intersection_matrix.clip(min=0.00001)
    )
    return intersection_matrix, union_matrix


def _binary_mask_inter_union(
    boxes1, boxes2, masks1, masks2, speed_up_matrix=None
):
    """
    Arguments
    ---------
        masks1 : [N1, H, W]
        masks2 : [N2, H, W]
        intersection_boxes : [4, N1, N2] can be passed for speedup

    Returns
    -------
        iou_matrix : [N1, N2]
    """
    assert len(masks1.shape) == 3 and len(masks2.shape) == 3

    masks1_area = torch.count_nonzero(masks1, axis=(-2, -1))
    masks2_area = torch.count_nonzero(masks2, axis=(-2, -1))
    (masks1_area_matrix, masks2_area_matrix) = torch.meshgrid(
        masks1_area, masks2_area
    )

    xywh1 = torchvision.ops.box_convert(boxes1, "xyxy", "xywh").int()
    if speed_up_matrix is None:
        speed_up_matrix = torchvision.ops.box_iou(boxes1, boxes2)

    intersection_matrix = []
    for row, (mask1, (x, y, w, h)) in enumerate(zip(masks1, xywh1)):
        intersection_row = torch.zeros((len(masks2)))
        notnull = torch.where(speed_up_matrix[row] > 0)[0]
        mask1_crop = TF.crop(mask1, y, x, h, w)
        masks2_crop = TF.crop(masks2[notnull], y, x, h, w)
        intersection_row[notnull] = torch.count_nonzero(
            mask1_crop * masks2_crop, axis=(-2, -1)
        ).float()

        intersection_matrix.append(intersection_row)

    intersection_matrix = torch.stack(intersection_matrix, axis=0)
    union_matrix = (
        masks1_area_matrix
        + masks2_area_matrix
        - intersection_matrix.clip(min=0.00001)
    )
    return intersection_matrix, union_matrix


def hybrid_iou(boxes1, boxes2, masks1, masks2, iou_threshold=0.5):
    """
    1. Compute box_iou for all box pairs
    2. For all box_iou values above iou_threshold, replace them with mask_iou
    """
    box_iou = torchvision.ops.box_iou(boxes1, boxes2)
    speed_up_matrix = torch.logical_and(
        box_iou > iou_threshold, box_iou < 1
    ).bool()

    nms_mode = (boxes1 == boxes2).all()

    if nms_mode:
        speed_up_matrix = torch.triu(speed_up_matrix, 1)

    mask_iou = binary_mask_iou(
        boxes1=boxes1,
        boxes2=boxes2,
        masks1=masks1,
        masks2=masks2,
        speed_up_matrix=speed_up_matrix,
    )
    if nms_mode:
        mask_iou = mask_iou + mask_iou.T

    return speed_up_matrix * mask_iou + ~speed_up_matrix * box_iou


def nms(cost_mat, threshold):
    keep_idxs = []
    for idx, row in enumerate(cost_mat):
        if idx == 0 or (row[keep_idxs] <= threshold).all():
            keep_idxs.append(idx)
    return torch.tensor(keep_idxs)


def nms_limit_group_size(
    cost_mat, cost_threshold=0.5, group_size_treshold=5, group_belong_cost=0.5
):
    keep_idxs = []
    for idx, row in enumerate(cost_mat):
        always_add_first = idx == 0
        nms_condition = (row[keep_idxs] < cost_threshold).all()
        current_group_size = (row[keep_idxs] > group_belong_cost).sum()
        can_extend_group = current_group_size < group_size_treshold
        if always_add_first or (nms_condition and can_extend_group):
            keep_idxs.append(idx)
    return torch.tensor(keep_idxs)


def box_nms_limit_group_size(
    boxes,
    scores,
    labels,
    iou_threshold=0.9,
    group_size_treshold=5,
    group_belong_cost=0.5,
):
    """
    This function performs the nms algorithm with hybrid iou as a distance metric
    """
    assert_ranking(scores)

    iou = torchvision.ops.box_iou(boxes, boxes)
    keep_idxs = []
    for class_id in torch.unique(labels):
        class_idxs = torch.where(class_id == labels)[0]
        class_iou = iou[class_idxs][:, class_idxs].clone()
        keep_class_local_idxs = nms_limit_group_size(
            class_iou, iou_threshold, group_size_treshold, group_belong_cost
        )
        keep_class_idxs = class_idxs[keep_class_local_idxs]
        keep_idxs.append(keep_class_idxs)
    keep_idxs = torch.cat(keep_idxs)
    keep = keep_idxs[scores[keep_idxs].argsort(descending=True)]
    return keep


def assert_ranking(attribute):
    assert (
        attribute.argsort(descending=True) == torch.arange(len(attribute))
    ).all()


def box_nms(boxes, scores, labels, iou_threshold=0.5):
    """
    This function performs the nms algorithm with hybrid iou as a distance metric
    """
    assert_ranking(scores)

    iou = torchvision.ops.box_iou(boxes, boxes)
    keep_mask = torch.zeros(len(boxes), dtype=torch.bool)
    for class_id in torch.unique(labels):
        class_idxs = torch.where(class_id == labels)[0]

        class_iou = iou[class_idxs][:, class_idxs].clone()
        keep_class_local_idxs = nms(class_iou, iou_threshold)
        keep_mask[class_idxs[keep_class_local_idxs]] = True
    keep_idxs = torch.where(keep_mask)[0]
    keep = keep_idxs[scores[keep_idxs].argsort(descending=True)]
    return keep


def mask_nms(
    boxes,
    scores,
    labels,
    masks,
    hybrid_nms_iou_threshold=0.5,
    hybrid_iou_thresh=0.5,
):
    """
    This function performs the nms algorithm with hybrid iou as a distance metric
    """
    assert_ranking(scores)

    iou = hybrid_iou(
        boxes, boxes, masks, masks, iou_threshold=hybrid_iou_thresh
    )

    keep_idxs = []
    for class_id in torch.unique(labels):
        class_idxs = torch.where(class_id == labels)[0]
        class_iou = iou[class_idxs][:, class_idxs].clone()
        keep_class_local_idxs = nms(class_iou, hybrid_nms_iou_threshold)
        keep_class_idxs = class_idxs[keep_class_local_idxs]
        keep_idxs.append(keep_class_idxs)
    keep_idxs = torch.cat(keep_idxs)
    keep = keep_idxs[scores[keep_idxs].argsort(descending=True)]
    return keep


from torchvision.ops.boxes import _box_inter_union


def box_io_min_max(boxes1, boxes2):
    """
    intersection over minimum area
    """
    inter, _ = _box_inter_union(boxes1, boxes2)
    areas1 = torchvision.ops.box_area(boxes1)
    areas2 = torchvision.ops.box_area(boxes2)
    areas = torch.stack(
        [
            areas1.unsqueeze(1).repeat(1, len(areas2)),
            areas2.unsqueeze(0).repeat(len(areas1), 1),
        ],
        dim=0,
    ).sort(dim=0, descending=True)[0]
    io_min = inter / areas[1]
    io_max = inter / areas[0]
    return io_min, io_max


def mask_areas(masks):
    return masks.sum(dim=(-2, -1)).flatten()


def mask_io_min_max(boxes1, boxes2, masks1, masks2):
    """
    intersection over minimum area
    """
    box_iou = torchvision.ops.box_iou(boxes1, boxes2)
    speed_up_matrix = box_iou > 0.01
    inter, _ = _binary_mask_inter_union(
        boxes1=boxes1,
        boxes2=boxes2,
        masks1=masks1,
        masks2=masks2,
        speed_up_matrix=speed_up_matrix,
    )
    areas1 = mask_areas(masks1)
    areas2 = mask_areas(masks2)
    areas = torch.stack(
        [
            areas1.unsqueeze(1).repeat(1, len(areas2)),
            areas2.unsqueeze(0).repeat(len(areas1), 1),
        ],
        dim=0,
    ).sort(dim=0, descending=True)[0]
    io_min = inter / areas[1]
    io_max = inter / areas[0]
    return io_min, io_max


def box_area_nms(
    boxes, scores, labels, masks, box_area_nms_iomin_threshold=0.95,
):
    """
    in analogy to nms, this function supresses overlapping boxes, however ranked by area
    also, instead of intersection over union, intersection over min area is used
    this filters out false positives that are not catched by regular nms, because the iou is low
    this happens when for example the upper body is detected as if it was an extra object
    """
    assert_ranking(scores)

    areas = torchvision.ops.box_area(boxes)
    io_min, io_max = box_io_min_max(boxes, boxes)
    cost_matrix = io_min
    cost_matrix[io_max < 0.5] = 0

    keep_idxs = []
    for class_id in torch.unique(labels):
        class_idxs = torch.where(class_id == labels)[0]
        class_iou = cost_matrix[class_idxs][:, class_idxs].clone()
        keep_class_local_idxs = nms(class_iou, box_area_nms_iomin_threshold)
        keep_class_idxs = class_idxs[keep_class_local_idxs]
        keep_idxs.append(keep_class_idxs)
    keep_idxs = torch.cat(keep_idxs)
    keep = keep_idxs[scores[keep_idxs].argsort(descending=True)]
    return keep


def mask_area_nms(
    boxes,
    scores,
    labels,
    masks,
    mask_nms_iomin_thresh=0.9,
    mask_nms_iomax_thresh=0.5,
):
    """
    in analogy to nms, this function supresses overlapping boxes, however ranked by area
    also, instead of intersection over union, intersection over min area is used
    this filters out false positives that are not catched by regular nms, because the iou is low
    this happens when for example the upper body is detected as if it was an extra object
    """
    assert_ranking(scores)

    io_min, io_max = mask_io_min_max(
        boxes1=boxes, boxes2=boxes, masks1=masks, masks2=masks
    )
    cost_matrix = torch.logical_or(
        io_max > mask_nms_iomax_thresh, io_min > mask_nms_iomin_thresh
    ).float()

    keep_idxs = []
    for class_id in torch.unique(labels):
        class_idxs = torch.where(class_id == labels)[0]
        class_iou = cost_matrix[class_idxs][:, class_idxs].clone()
        keep_class_local_idxs = nms(class_iou, 0.5)
        keep_class_idxs = class_idxs[keep_class_local_idxs]

        keep_idxs.append(keep_class_idxs)
    keep_idxs = torch.cat(keep_idxs)
    keep = keep_idxs[scores[keep_idxs].argsort(descending=True)]
    return keep


def keep_low_score_only_if_occluded(
    boxes, scores, vis_thresh=0.5, high_score_thresh=0.5
):
    iou = torchvision.ops.box_iou(boxes, boxes)
    remove_diag = ~torch.eye((len(iou))).bool()
    high_score = scores > high_score_thresh
    occluded = (iou * remove_diag > vis_thresh)[:, high_score].any(dim=1)
    keep = torch.logical_or(
        high_score, torch.logical_and(~high_score, occluded)
    )
    return keep


def keep_keypoint_containing_detections(
    boxes,
    scores,
    keypoint_idxs,
    keypoints_scores,
    keypoints_percentage_thresh=0.5,
    keypoints_score_thresh=0.5,
    vis_thresh=0.25,
    high_score_thresh=0.5,
):
    """
    keypoint_idxs : list of indices of required keypoints
    keypoints_percentage_thresh : percentage how many of those need to be found to keep object
    vis_thresh : minimum visibility of objects that are scanned for keypoints and possibly removed
    """
    iou = torchvision.ops.box_iou(boxes, boxes)
    remove_diag = ~torch.eye((len(iou))).bool()
    high_score = scores > high_score_thresh
    visible = (iou * remove_diag < vis_thresh)[:, high_score].all(dim=1)
    # large_enough = torchvision.ops.box_area(boxes) > 1500

    relevant_values = keypoints_scores[:, keypoint_idxs]
    num_found_relevant_keypoints = (
        relevant_values > keypoints_score_thresh
    ).sum(dim=1)
    num_required_keypoints = keypoints_percentage_thresh * len(keypoint_idxs)
    keypoints_found = num_found_relevant_keypoints > num_required_keypoints
    remove = torch.logical_and(
        ~high_score, torch.logical_and(visible, ~keypoints_found)
    )
    keep = ~remove
    return keep


def keypoint_convert(keys, inp_fmt, out_fmt):
    if inp_fmt == "name" and out_fmt == "idx":
        return [keypoint_name2idx[name] for name in keys]
    if inp_fmt == "idx" and out_fmt == "name":
        return [keypoint_idx2name[idx] for idx in keys]
