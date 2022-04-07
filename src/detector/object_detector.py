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

from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import resnet50
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FasterRCNN


def init_faster_rcnn(
    num_classes=2,
    nms_thresh=0.5,
    score_thresh=0.5,
    detections_per_img=100,
    pretrained=True,
):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.detections_per_img = detections_per_img
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.return_segmentation = False
    return model


def init_mask_rcnn(
    num_classes=2,
    nms_thresh=0.5,
    score_thresh=0.5,
    mask_score_thresh=0.5,
    detections_per_img=100,
    pretrained=True,
):
    original_maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained
    )
    model = MyMaskRCNN(backbone=original_maskrcnn.backbone, num_classes=91)
    model.load_state_dict(original_maskrcnn.state_dict())
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.detections_per_img = detections_per_img
    model.roi_heads.score_thresh = score_thresh
    model.mask_score_thresh = mask_score_thresh

    if num_classes != 91:
        print("\ninit new heads")
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, num_classes=num_classes
        )
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=256,
            num_classes=num_classes,
        )
    model.return_segmentation = True
    return model


def init_detector_from_config(hyperparams):
    if hyperparams["return_segmentation"]:
        return init_mask_rcnn(
            num_classes=hyperparams["num_classes"],
            score_thresh=hyperparams["score_thresh"],
            mask_score_thresh=hyperparams["mask_score_thresh"],
            nms_thresh=hyperparams["nms_thresh"],
            detections_per_img=hyperparams["detections_per_img"],
            pretrained=hyperparams["pretrained"],
        )

    else:
        return init_faster_rcnn(
            num_classes=hyperparams["num_classes"],
            score_thresh=hyperparams["score_thresh"],
            nms_thresh=hyperparams["nms_thresh"],
            detections_per_img=hyperparams["detections_per_img"],
            pretrained=hyperparams["pretrained"],
        )


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
    def __init__(self, backbone, num_classes, mask_score_thresh=0.5, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)
        self.mask_score_thresh = mask_score_thresh
        self.min_mask_pixels = 20

    def forward(self, images, targets=None):
        result = super().forward(images, targets)
        if self.training:
            return result
        else:
            detection_batch = result
            keep_detection_batch = []
            for detection in detection_batch:
                detection["masks"] = (
                    detection["masks"] > self.mask_score_thresh
                ).int()
                num_pixels_per_mask = detection["masks"].sum((1, 2, 3))
                keep = num_pixels_per_mask > self.min_mask_pixels
                keep_detection = {}
                for key, data in detection.items():
                    keep_detection[key] = data[keep]
                keep_detection_batch.append(keep_detection)
            return keep_detection_batch

