from turtle import forward
from requests import post
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN, MaskRCNN, KeypointRCNN
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    mobilenet_backbone,
)
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

from torch import det, nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
)
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
from src.detector.utils import box_nms_limit_group_size
from src.detector.utils import mask_nms, box_area_nms, mask_area_nms
from src.detector.utils import (
    keep_keypoint_containing_detections,
    keypoint_convert,
)
from src.detector.utils import keep_low_score_only_if_occluded


body_part_combination = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "arms": ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
    "legs": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
    "feet": ["left_ankle", "right_ankle"],
    "left_extremity": ["left_knee", "left_elbow", "left_ankle", "left_wrist"],
    "right_extremity": [
        "right_knee",
        "right_elbow",
        "right_ankle",
        "right_wrist",
    ],
    "left_torso": ["left_shoulder", "left_hip"],
    "right_torso": ["right_shoulder", "right_hip"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "extremity": [
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
}


def build_backbone(model):
    return model.backbone


def build_region_proposal_network(model):
    return model.rpn


def build_anchor_generator(anchor_sizes, anchor_aspect_ratios):
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    anchor_sizes = tuple([(size,) for size in anchor_sizes])
    aspect_ratios = (tuple([ratio for ratio in anchor_aspect_ratios]),) * len(
        anchor_sizes
    )
    return AnchorGenerator(anchor_sizes, aspect_ratios)


def build_box_heads(
    faster_rcnn, train=False, num_classes=91, return_none=False
):
    if return_none:
        return {"predictor": None, "head": None, "roi_pool": None}

    if train:
        box_predictor = FastRCNNPredictor(
            in_channels=faster_rcnn.roi_heads.box_predictor.cls_score.in_features,
            num_classes=num_classes,
        )
    else:
        box_predictor = faster_rcnn.roi_heads.box_predictor

    box_head = faster_rcnn.roi_heads.box_head
    box_roi_pool = faster_rcnn.roi_heads.box_roi_pool
    return {
        "predictor": box_predictor,
        "head": box_head,
        "roi_pool": box_roi_pool,
    }


def build_mask_heads(mask_rcnn, train=False, num_classes=91, return_none=False):
    if return_none:
        return {"predictor": None, "head": None, "roi_pool": None}
    if train:
        mask_predictor = MaskRCNNPredictor(
            in_channels=mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels,
            dim_reduced=256,
            num_classes=num_classes,
        )
    else:
        mask_predictor = mask_rcnn.roi_heads.mask_predictor
    mask_head = mask_rcnn.roi_heads.mask_head
    mask_roi_pool = mask_rcnn.roi_heads.mask_roi_pool
    return {
        "predictor": mask_predictor,
        "head": mask_head,
        "roi_pool": mask_roi_pool,
    }


def build_keypoint_heads(keypoint_rcnn, train=False, return_none=False):
    if return_none:
        return {"predictor": None, "head": None, "roi_pool": None}

    keypoint_predictor = keypoint_rcnn.roi_heads.keypoint_predictor
    keypoint_head = keypoint_rcnn.roi_heads.keypoint_head
    keypoint_roi_pool = keypoint_rcnn.roi_heads.keypoint_roi_pool
    return {
        "predictor": keypoint_predictor,
        "head": keypoint_head,
        "roi_pool": keypoint_roi_pool,
    }


def init_detector(
    num_classes=91,
    nms_thresh=0.5,
    entropy_thresh=0.05,
    low_score_thresh=0.5,
    mask_score_thresh=0.5,
    detections_per_img=100,
    return_segmentation=False,
    use_mask_nms=False,
    mask_nms_iomax_thresh=0.5,
    mask_nms_iomin_thresh=0.9,
    anchor_sizes=[32, 64, 128, 256, 512],
    anchor_aspect_ratios=[2.0],
    nms_group_size_treshold=3,
    nms_group_belong_cost=0.9,
    return_keypoints=False,
    use_keypoint_filter=False,
    use_low_score_occlusion_filter=False,
    keypoint_body_parts="extremity",
    keypoints_coverage_thresh=0.5,
    low_score_occlusion_thresh=0.0,
    high_score_thresh=0.5,
    keypoints_score_thresh=0.5,
    **kwargs
):

    # PREPARE PRETRAINED PYTORCH COMPONENTS
    ###################################################
    mask_rcnn = maskrcnn_resnet50_fpn(pretrained=True)
    keypoint_rcnn = keypointrcnn_resnet50_fpn(pretrained=True)

    backbone = build_backbone(mask_rcnn)
    rpn = build_region_proposal_network(mask_rcnn)

    # BUILD PREFILTER
    ###################################################
    prefilter = GroupNmsPrefilter(
        entropy_thresh=entropy_thresh,
        nms_group_size_treshold=nms_group_size_treshold,
        nms_group_belong_cost=nms_group_belong_cost,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        score_thresh=low_score_thresh,
    )

    # BUILD POSTFILTER
    ###################################################
    # in an ideal scenario postfilter gets all the necessary inputs from the detector output
    # however pretrained keypointrcnn heads require a different feature map than pretrained maskrcnn heads
    # you could fix it by training them jointly, but I dont have the time for that
    # quick fix is run keypointrcnn on proposed boxes from maskrcnn, but feature map needs to be computed twice

    postfilter = Postfilter(
        use_mask_nms=use_mask_nms,
        mask_nms_iomax_thresh=mask_nms_iomax_thresh,
        mask_nms_iomin_thresh=mask_nms_iomin_thresh,
        mask_score_thresh=mask_score_thresh,
        use_keypoint_filter=use_keypoint_filter,
        keypoint_body_parts=keypoint_body_parts,
        keypoints_coverage_thresh=keypoints_coverage_thresh,
        low_score_occlusion_thresh=low_score_occlusion_thresh,
        use_low_score_occlusion_filter=use_low_score_occlusion_filter,
        high_score_thresh=high_score_thresh,
        keypoints_score_thresh=keypoints_score_thresh,
    )

    if use_keypoint_filter:
        keypoint_heads = build_keypoint_heads(keypoint_rcnn, return_none=False)
        external_keypoint_roi_heads = PrefilterRoiHeads(
            prefilter=lambda x: x,
            keypoint_predictor=keypoint_heads["predictor"],
            keypoint_head=keypoint_heads["head"],
            keypoint_roi_pool=keypoint_heads["roi_pool"],
        )
        external_keypointrcnn = CustomRCNN(
            backbone=build_backbone(keypoint_rcnn),
            rpn=None,
            roi_heads=external_keypoint_roi_heads,
            postfilter=lambda detection_batch, images: detection_batch,
        )
        postfilter.add_external_component(keypointrcnn=external_keypointrcnn)

    # Model Roi Heads
    #########################################################
    box_heads = build_box_heads(mask_rcnn, train=num_classes != 91)
    mask_heads = build_mask_heads(
        mask_rcnn, train=num_classes != 91, return_none=not return_segmentation
    )

    roi_heads = PrefilterRoiHeads(
        prefilter=prefilter,
        box_predictor=box_heads["predictor"],
        box_head=box_heads["head"],
        box_roi_pool=box_heads["roi_pool"],
        mask_predictor=mask_heads["predictor"],
        mask_head=mask_heads["head"],
        mask_roi_pool=mask_heads["roi_pool"],
    )

    # Model
    ########################################################
    model = CustomRCNN(
        backbone=backbone, rpn=rpn, roi_heads=roi_heads, postfilter=postfilter
    )
    return model


class GroupNmsPrefilter:
    def __init__(
        self,
        entropy_thresh,
        score_thresh,
        nms_thresh,
        nms_group_size_treshold,
        nms_group_belong_cost,
        detections_per_img,
    ):
        self.score_thresh = score_thresh
        self.entropy_thresh = entropy_thresh
        self.nms_thresh = nms_thresh
        self.nms_group_size_treshold = nms_group_size_treshold
        self.nms_group_belong_cost = nms_group_belong_cost
        self.detections_per_img = detections_per_img

    def __call__(self, boxes, scores, labels):

        from scipy.stats import entropy
        import numpy as np

        # only keep low entropy predictions
        num_classes = 90
        h = torch.from_numpy(entropy(scores.reshape(-1, num_classes), axis=1))
        keep = h < self.entropy_thresh
        keep = keep.repeat_interleave(num_classes)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # only keep high_score predictions
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per clas
        rank = scores.argsort(descending=True)
        boxes, scores, labels = boxes[rank], scores[rank], labels[rank]
        keep = box_nms_limit_group_size(
            boxes,
            scores,
            labels,
            iou_threshold=self.nms_thresh,
            group_size_treshold=self.nms_group_size_treshold,
            group_belong_cost=self.nms_group_belong_cost,
        )

        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        return boxes, scores, labels


from torchvision.models.detection.roi_heads import (
    maskrcnn_inference,
    keypointrcnn_inference,
)


class DefaultRoIHeads(RoIHeads):
    """
    This is an Empty RoiHeads class, which does sets all the hyperparameters to the default values of the RoiHeads in FasterRCNN.
    If you subclass DefaultRoIHeads, you can initialize RoIHeads using only the necessary building blocks, without any hyperparameters.

    """

    def __init__(
        self, box_predictor=None, box_head=None, box_roi_pool=None, **kwargs
    ):
        super().__init__(
            box_predictor=box_predictor,
            box_head=box_head,
            box_roi_pool=box_roi_pool,
            score_thresh=None,
            nms_thresh=None,
            detections_per_img=None,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            **kwargs
        )


class ExternalCapabilityRoiHeads(DefaultRoIHeads):
    """
    This version of RoiHeads can be run just like the original one.
    But it also lets you choose to be apply all the remaining heads besides the box head on the output of an external detector.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def has_keypoint(self):
        return (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        )

    def run_box_head(self, features, result, proposals, image_shapes):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {"boxes": boxes[i], "labels": labels[i], "scores": scores[i],}
            )
        return result

    def run_keypoint_head(self, features, result, image_shapes):
        keypoint_proposals = [p["boxes"] for p in result]
        keypoint_features = self.keypoint_roi_pool(
            features, keypoint_proposals, image_shapes
        )
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)
        keypoints_probs, kp_scores = keypointrcnn_inference(
            keypoint_logits, keypoint_proposals
        )
        for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
            r["keypoints"] = keypoint_prob
            r["keypoints_scores"] = kps
        return result

    def run_mask_head(self, features, result, image_shapes):
        mask_proposals = [p["boxes"] for p in result]
        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(
                features, mask_proposals, image_shapes
            )
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
        labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, r in zip(masks_probs, result):
            r["masks"] = mask_prob
        return result

    def forward(
        self,
        features,
        proposals=None,
        image_shapes=None,
        targets=None,
        result=None,
    ):
        losses = {}

        assert proposals is None or result is None

        if not proposals is None:
            result: List[Dict[str, torch.Tensor]] = []
            result = self.run_box_head(
                features, result, proposals, image_shapes
            )

        if self.has_mask():
            result = self.run_mask_head(features, result, image_shapes)

        if self.has_keypoint():
            result = self.run_keypoint_head(features, result, image_shapes)

        return result, losses


class PrefilterRoiHeads(ExternalCapabilityRoiHeads):
    """
    This version of RoiHeads allows you to customize the postprocessing of detections by adding a custom filter object.
    Don't worry about the postprocess_detections method. It is only written here, so that you can apply self.prefilter.
    Also, don't be confused by the nomenclature. Pytorch source code calls this 'postprocess_detections', however I use the name 
    'pre-filter' inside this method. This is because I have a two stage filtering process, which also includes a 'postfilter' later down the line.
    """

    def __init__(self, prefilter, **kwargs):
        super().__init__(**kwargs)
        self.prefilter = prefilter

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes,
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [
            boxes_in_image.shape[0] for boxes_in_image in proposals
        ]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)

            boxes, scores, labels = self.prefilter(boxes, scores, labels)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels


class Postfilter(nn.Module):
    def __init__(
        self,
        mask_score_thresh=0.5,
        min_mask_pixels=20,
        use_mask_nms=False,
        mask_nms_iomax_thresh=0.5,
        mask_nms_iomin_thresh=0.9,
        use_keypoint_filter=False,
        use_low_score_occlusion_filter=False,
        low_score_occlusion_thresh=0.1,
        high_score_thresh=0.5,
        keypoint_body_parts="extremity",
        keypoints_coverage_thresh=0.5,
        keypoints_score_thresh=0.5,
    ):
        super().__init__()
        # deal with small masks
        self.mask_score_thresh = mask_score_thresh
        self.min_mask_pixels = min_mask_pixels

        # mask nms
        self.use_mask_nms = use_mask_nms
        self.mask_nms_iomin_thresh = mask_nms_iomin_thresh
        self.mask_nms_iomax_thresh = mask_nms_iomax_thresh

        # use_low_score_occlusion_filter
        self.high_score_thresh = high_score_thresh
        self.use_low_score_occlusion_filter = use_low_score_occlusion_filter

        # keypoint filtering
        self.use_keypoint_filter = use_keypoint_filter
        self.keypoint_body_parts = keypoint_body_parts
        self.keypoints_coverage_thresh = keypoints_coverage_thresh
        self.low_score_occlusion_thresh = low_score_occlusion_thresh
        self.keypoints_score_thresh = keypoints_score_thresh

        self.external_components = nn.ModuleDict({})

    def add_external_component(self, **kwargs):
        self.external_components.update(kwargs)

    def __call__(self, detection_batch, images):
        detection_batch = remove_small_masks_from_det(
            detection_batch,
            mask_score_thresh=self.mask_score_thresh,
            min_mask_pixels=self.min_mask_pixels,
        )
        if self.use_mask_nms:
            detection_batch = apply_mask_nms_on_det(
                detection_batch,
                mask_nms_iomin_thresh=self.mask_nms_iomin_thresh,
                mask_nms_iomax_thresh=self.mask_nms_iomax_thresh,
            )

        detection_batch = extract_pedestrians_from_det(detection_batch)
        if self.use_keypoint_filter:
            if "keypoints" not in detection_batch[0].keys():
                detection_batch = self.external_components["keypointrcnn"](
                    detections=detection_batch, images=images
                )

            detection_batch = apply_keypoint_filtering(
                detection_batch,
                keypoint_names=body_part_combination[self.keypoint_body_parts],
                keypoints_percentage_thresh=self.keypoints_coverage_thresh,
                high_score_thresh=self.high_score_thresh,
                low_score_occlusion_thresh=self.low_score_occlusion_thresh,
                keypoints_score_thresh=self.keypoints_score_thresh,
                images=images,
            )
        if self.use_low_score_occlusion_filter:
            detection_batch = apply_score_filtering_on_area(
                detection_batch,
                high_score_thresh=self.high_score_thresh,
                low_score_occlusion_thresh=self.low_score_occlusion_thresh,
            )
        return detection_batch


def get_image_scale_value(image, transform):
    return min(
        transform.min_size[-1] / min(list(image[0].shape)),
        transform.max_size / max(list(image[0].shape)),
    )


class CustomRCNN(FasterRCNN):
    """
    Changes from original pytorch GeneralizedRCNN

    - you can run this detector on external region proposals (and even on external detections, to add masks and or keypoints)
    - you can add a custom postprocessing filter, that uses custom logic to filter out detections
    """

    def __init__(
        self,
        backbone,
        roi_heads=None,
        rpn=None,
        postfilter=lambda detection_batch, images: detection_batch,
        num_classes=91,
        **kwargs
    ):
        super().__init__(backbone=backbone, num_classes=num_classes, **kwargs)
        if not rpn is None:
            self.rpn = rpn
        if not roi_heads is None:
            self.roi_heads = roi_heads
        self.postfilter = postfilter

    def forward(self, images, targets=None, proposals=None, detections=None):
        """
        You can run CustomRCNN in 3 different modes:

        1) Full Pipeline 
            (proposals=None, detections=None)

        2) Apply all heads on external region proposals 
            (proposals : List[Tensor[N, 4]], detections=None)

        3) Apply all heads besides box-regression/box-classification on external boxes 
            (proposals=None, detections : List[
                                            dict{
                                                "boxes": Tensor[N, 4], 
                                                "scores": Tensor[N], 
                                                "labels": Tensor[N]
                                                }
                                            ])
        
        """
        if self.training:
            losses = super().forward(images, targets)
            return losses
        else:
            assert not (not proposals is None and not detections is None)
            if not proposals is None:
                detection_batch = self.predict_on_external_proposals(
                    images, proposals
                )
            elif not detections is None:
                detection_batch = self.predict_on_external_detections(
                    images, detections
                )
            else:
                detection_batch = super().forward(images)

            return self.postfilter(detection_batch, images=images)

    def predict_on_external_proposals(
        self, images, proposals=None, targets=None
    ):
        original_image_sizes: List[Tuple[int, int]] = []
        scaled_proposals = []
        for img, img_proposals in zip(images, proposals):
            scale = get_image_scale_value(img, self.transform)
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
            scaled_proposals.append(img_proposals * scale)

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        detections, detector_losses = self.roi_heads(
            features=features,
            proposals=scaled_proposals,
            image_shapes=images.image_sizes,
            targets=targets,
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )
        return self.eager_outputs(detector_losses, detections)

    def predict_on_external_detections(self, images, detections, targets=None):
        original_image_sizes: List[Tuple[int, int]] = []
        roi_heads_input = []
        for img, img_det in zip(images, detections):
            scale = get_image_scale_value(img, self.transform)
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
            scaled_img_det = {
                "boxes": scale * img_det["boxes"].clone(),
                "scores": img_det["scores"].clone(),
                "labels": img_det["labels"].clone(),
            }
            roi_heads_input.append(scaled_img_det)

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        detections_update, detector_losses = self.roi_heads(
            features=features,
            result=roi_heads_input,
            image_shapes=images.image_sizes,
            targets=targets,
        )
        detections_update = self.transform.postprocess(
            detections_update, images.image_sizes, original_image_sizes
        )
        for det, update in zip(detections, detections_update):
            det.update(update)
        return self.eager_outputs(detector_losses, detections)


def extract_pedestrians_from_det(detection_batch):
    keep_detection_batch = []
    for detection in detection_batch:
        keep = detection["labels"] == 1
        keep_detection = {key: data[keep] for (key, data) in detection.items()}
        keep_detection_batch.append(keep_detection)
    return keep_detection_batch


def remove_body_part_positives_from_det(
    detection_batch, mask_nms_iomax_thresh=0.5, mask_nms_iomin_thresh=0.9
):
    keep_detection_batch = []
    for detection in detection_batch:
        keep = mask_area_nms(
            boxes=detection["boxes"],
            scores=detection["scores"],
            labels=detection["labels"],
            masks=detection["masks"].squeeze(1),
            mask_nms_iomin_thresh=mask_nms_iomin_thresh,
            mask_nms_iomax_thresh=mask_nms_iomax_thresh,
        )
        keep_detection = {key: data[keep] for (key, data) in detection.items()}
        keep_detection_batch.append(keep_detection)
    return keep_detection_batch


def apply_score_filtering_on_area(
    detection_batch, high_score_thresh=0.5, low_score_occlusion_thresh=0.5
):
    keep_detection_batch = []
    for detection in detection_batch:
        keep = keep_low_score_only_if_occluded(
            boxes=detection["boxes"],
            scores=detection["scores"],
            high_score_thresh=high_score_thresh,
            vis_thresh=low_score_occlusion_thresh,
        )
        keep_detection = {key: data[keep] for (key, data) in detection.items()}
        keep_detection_batch.append(keep_detection)
    return keep_detection_batch


def apply_mask_nms_on_det(
    detection_batch, mask_nms_iomin_thresh=0.9, mask_nms_iomax_thresh=0.5
):
    keep_detection_batch = []
    for detection in detection_batch:
        keep = mask_area_nms(
            boxes=detection["boxes"],
            scores=detection["scores"],
            labels=detection["labels"],
            masks=detection["masks"].squeeze(1),
            mask_nms_iomax_thresh=mask_nms_iomax_thresh,
            mask_nms_iomin_thresh=mask_nms_iomin_thresh,
        )
        keep_detection = {key: data[keep] for (key, data) in detection.items()}

        keep_detection_batch.append(keep_detection)
    return keep_detection_batch


def remove_small_masks_from_det(
    detection_batch, mask_score_thresh=0.5, min_mask_pixels=20,
):
    keep_detection_batch = []
    for detection in detection_batch:
        detection["masks"] = (detection["masks"] > mask_score_thresh).float()
        num_pixels_per_mask = detection["masks"].sum((1, 2, 3))
        keep = num_pixels_per_mask > min_mask_pixels
        keep_detection = {key: data[keep] for (key, data) in detection.items()}
        keep_detection_batch.append(keep_detection)
    return keep_detection_batch


def apply_keypoint_filtering(
    detection_batch,
    keypoint_names,
    keypoints_percentage_thresh=0.5,
    high_score_thresh=0.5,
    low_score_occlusion_thresh=0.1,
    keypoints_score_thresh=0.5,
    images=None,
):
    keep_detection_batch = []
    for detection in detection_batch:
        keep = keep_keypoint_containing_detections(
            boxes=detection["boxes"],
            scores=detection["scores"],
            keypoint_idxs=keypoint_convert(keypoint_names, "name", "idx"),
            keypoints_scores=torch.sigmoid(detection["keypoints_scores"]),
            keypoints_percentage_thresh=keypoints_percentage_thresh,
            keypoints_score_thresh=keypoints_score_thresh,
            vis_thresh=low_score_occlusion_thresh,
            high_score_thresh=high_score_thresh,
        )
        keep_detection = {key: data[keep] for (key, data) in detection.items()}
        keep_detection_batch.append(keep_detection)
    return keep_detection_batch


def correct_box_sizes_using_keypoints(
    detection_batch,
    keypoint_score_thresh=0.9,
    scale_factor_width=1.1,
    scale_factor_height=1.1,
):
    for det in detection_batch:
        le_idxs = keypoint_convert(
            body_part_combination["left_extremity"], "name", "idx"
        )
        re_idxs = keypoint_convert(
            body_part_combination["right_extremity"], "name", "idx"
        )
        # lt_idxs = keypoint_convert(b["left_torso"] , "name", "idx")
        # rt_idxs = keypoint_convert(b["right_torso"] , "name", "idx")
        f_idxs = keypoint_convert(body_part_combination["feet"], "name", "idx")
        xywh = torchvision.ops.box_convert(det["boxes"].clone(), "xyxy", "xywh")

        for obj_idx in range(len(det["keypoints_scores"])):
            found_keypoint_idxs = torch.where(
                det["keypoints_scores"][obj_idx].sigmoid()
                > keypoint_score_thresh
            )[0].tolist()
            if set(f_idxs) == (set(f_idxs) - set(found_keypoint_idxs)):
                xywh[obj_idx, 3] *= scale_factor_height

            if len((set(re_idxs) - set(found_keypoint_idxs))) > 0:
                xywh[obj_idx, 0] -= (scale_factor_width - 1) * xywh[obj_idx, 2]

            if len((set(le_idxs) - set(found_keypoint_idxs))) > 0:
                xywh[obj_idx, 2] *= scale_factor_width
        boxes = torchvision.ops.box_convert(xywh, "xywh", "xyxy")
        det["boxes"] = boxes
    return detection_batch

