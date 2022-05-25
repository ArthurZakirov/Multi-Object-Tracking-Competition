import torch
import torchvision

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
            "facebookresearch/detr:main", "detr_resnet50", pretrained=True
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
        det = super().forward(images)
        frcnn_format_dets = []
        for (image_logits, image_boxes) in zip(
            det["pred_logits"], det["pred_boxes"]
        ):
            logits = F.softmax(image_logits, dim=-1)
            scores, labels = logits.max(dim=-1)
            width = 1920
            height = 1080
            image_boxes[:, [0, 2]] *= width
            image_boxes[:, [1, 3]] *= height
            image_boxes = torchvision.ops.box_convert(
                image_boxes, "cxcywh", "xyxy"
            )
            not_background = torch.logical_and(labels != 91, self.score_thresh)

            frcnn_format_det = {
                "scores": scores[not_background],
                "labels": labels[not_background],
                "boxes": image_boxes[not_background],
            }
            frcnn_format_dets.append(frcnn_format_det)
        retur frcnn_format_dets # TODO : Postprocessing