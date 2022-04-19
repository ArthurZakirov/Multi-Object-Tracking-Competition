from turtle import forward
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    mobilenet_backbone,
)
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
from types import MethodType

from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import resnet50
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FasterRCNN

from src.detector.utils import mask_nms


def init_detector(
    num_classes=91,
    nms_thresh=0.5,
    score_thresh=0.5,
    mask_score_thresh=0.5,
    detections_per_img=100,
    pretrained=True,
    return_segmentation=False,
    use_mask_nms=False,
    **kwargs
):
    if return_segmentation:
        model = MyMaskRCNN.from_original_maskrcnn(
            pretrained=pretrained,
            use_mask_nms=use_mask_nms,
            mask_score_thresh=mask_score_thresh,
        )

    else:
        original_fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained
        )
        model = original_fasterrcnn

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.detections_per_img = detections_per_img
    model.roi_heads.score_thresh = score_thresh

    if num_classes != 91:
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, num_classes=num_classes
        )

        if return_segmentation:
            in_features_mask = (
                model.roi_heads.mask_predictor.conv5_mask.in_channels
            )
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_channels=in_features_mask,
                dim_reduced=256,
                num_classes=num_classes,
            )
    return model


class FRCNN_FPN(FasterRCNN):
    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone("resnet50", False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return (
            detections["boxes"].detach().cpu(),
            detections["scores"].detach().cpu(),
        )


class MyMaskRCNN(MaskRCNN):
    def __init__(
        self,
        backbone,
        num_classes=91,
        mask_score_thresh=0.5,
        min_mask_pixels=20,
        use_mask_nms=False,
        **kwargs
    ):
        super().__init__(backbone, num_classes, **kwargs)
        self.mask_score_thresh = mask_score_thresh
        self.min_mask_pixels = min_mask_pixels
        self.use_mask_nms = use_mask_nms

    @classmethod
    def from_original_maskrcnn(
        cls, pretrained, use_mask_nms, mask_score_thresh
    ):
        original_maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained
        )
        self = cls.__new__(cls)
        self.__init__(
            backbone=original_maskrcnn.backbone,
            use_mask_nms=use_mask_nms,
            mask_score_thresh=mask_score_thresh,
        )
        self.load_state_dict(original_maskrcnn.state_dict())
        return self

    def predict_on_external_proposals(self, images, proposals, targets=None):
        return run_fasterrcnn_on_external_proposals(
            self, images, proposals, targets
        )

    def forward(self, images, targets=None):
        if self.training:
            losses = super().forward(images, targets)
            return losses
        else:
            detection_batch = super().forward(images)
            return postprocess_mask_predictions(
                detection_batch,
                mask_score_thresh=self.mask_score_thresh,
                min_mask_pixels=self.min_mask_pixels,
                use_mask_nms=self.use_mask_nms,
            )


def postprocess_mask_predictions(
    detection_batch,
    mask_score_thresh=0.5,
    min_mask_pixels=20,
    fill_small_boxes=False,
    use_mask_nms=False,
):
    # remove detections with small masks
    if not fill_small_boxes:
        keep_detection_batch = []
        for detection in detection_batch:
            detection["masks"] = (detection["masks"] > mask_score_thresh).int()
            num_pixels_per_mask = detection["masks"].sum((1, 2, 3))
            keep = num_pixels_per_mask > min_mask_pixels
            keep_detection = {
                key: data[keep] for (key, data) in detection.items()
            }
            keep_detection_batch.append(keep_detection)
        detection_batch = keep_detection_batch

    # perform mask nms
    if use_mask_nms:
        keep_detection_batch = []
        for detection in detection_batch:
            keep = mask_nms(
                boxes=detection["boxes"],
                scores=detection["scores"],
                labels=detection["labels"],
                masks=detection["masks"].squeeze(1),
            )
            keep_detection = {
                key: data[keep] for (key, data) in detection.items()
            }
            keep_detection_batch.append(keep_detection)
        detection_batch = keep_detection_batch
    return detection_batch


def get_image_scale_value(image, transform):
    return min(
        transform.min_size[-1] / min(list(image[0].shape)),
        transform.max_size / max(list(image[0].shape)),
    )


def run_fasterrcnn_on_external_proposals(self, images, proposals, targets=None):
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
        features, scaled_proposals, images.image_sizes, targets
    )
    detections = self.transform.postprocess(
        detections, images.image_sizes, original_image_sizes
    )
    if "masks" in detections[0].keys():
        detections = postprocess_mask_predictions(
            detections,
            mask_score_thresh=self.mask_score_thresh,
            min_mask_pixels=self.min_mask_pixels,
            use_mask_nms=self.use_mask_nms,
        )
    return self.eager_outputs(detector_losses, detections)


original_detr = torch.hub.load(
    "facebookresearch/detr:main", "detr_resnet50", pretrained=False
)
HUBDETR = type(original_detr)


class DETR(HUBDETR):
    def __init__(
        self, backbone=None, transformer=None, num_classes=91, num_queries=100
    ):
        super().__init__(backbone, transformer, num_classes, num_queries)

    @classmethod
    def from_facebook_original(cls):
        original_detr = torch.hub.load(
            "facebookresearch/detr:main", "detr_resnet50", pretrained=False
        )
        self = cls.__new__(cls)
        self.__init__(
            backbone=original_detr.backbone,
            transformer=original_detr.transformer,
            num_classes=original_detr.class_embed.out_features - 1,
            num_queries=original_detr.num_queries,
        )
        self.load_state_dict(original_detr.state_dict())
        return self

    def forward(self, images):
        det = super()(images)
        frcnn_format_dets = []
        for (image_logits, image_boxes) in zip(
            det["pred_logits"], det["pred_boxes"]
        ):
            logits = F.softmax(image_logits, dim=-1)
            scores, labels = logits.max(dim=-1)
            image_boxes[:, [0, 2]] *= 1920
            image_boxes[:, [1, 3]] *= 1080
            image_boxes = torchvision.ops.box_convert(
                image_boxes, "cxcywh", "xyxy"
            )
            frcnn_format_det = {
                "scores": scores,
                "labels": labels,
                "boxes": image_boxes,
            }
            frcnn_format_dets.append(frcnn_format_det)
        return frcnn_format_dets

