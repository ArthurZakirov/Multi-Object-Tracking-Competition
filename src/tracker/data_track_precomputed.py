from msilib import sequence
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from src.detector.utils import decode_segmentation, mask_convert
from src.utils.file_utils import listdir_nohidden
from src.tracker.data_track import MOT16, MOT16Sequence, split_sequence_names

_sets = {}
splits = [
    "mini",
    "train",
    "test",
    "all",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "reid",
    "train_wo_val",
    "train_wo_val2",
    "val",
    "val2",
]

# Fill all available datasets, change here to modify / add new datasets.
for split in splits:
    dataset_name = f"MOT16-{split}"
    _sets[dataset_name] = lambda root_dir, split=split, **kwargs: MOT16(
        root_dir, split, **kwargs
    )


class MOT16SequencesPrecomputed(torch.utils.data.Dataset):
    def __init__(
        self,
        precomputed_data_root_dir,
        original_data_root_dir,
        split,
        vis_threshold=0.0,
        reid_on_det_model_name=None,
        reid_on_gt_model_name=None,
        return_det_segmentation=False,
        return_gt_segmentation=False,
        only_obj_w_mask=True,
    ):
        super().__init__()
        self._sequences = []
        sequence_names = split_sequence_names(split, original_data_root_dir)

        for seq_name in sequence_names:
            precomputed_seq_dir = os.path.join(
                precomputed_data_root_dir, seq_name
            )
            sequence = MOT16SequencePrecomputed(
                seq_name,
                original_data_root_dir,
                precomputed_seq_dir,
                vis_threshold,
                reid_on_det_model_name,
                reid_on_gt_model_name,
                return_det_segmentation,
                return_gt_segmentation,
                only_obj_w_mask,
            )
            self._sequences.append(sequence)

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        return self._sequences[idx]


class MOT16SequencePrecomputed(MOT16Sequence):
    def __init__(
        self,
        seq_name,
        original_data_root_dir,
        precomputed_seq_dir,
        vis_threshold=0.0,
        reid_on_det_model_name=None,
        reid_on_gt_model_name=None,
        return_det_segmentation=False,
        return_gt_segmentation=False,
        only_obj_w_mask=True,
    ):
        super().__init__(
            seq_name=seq_name,
            vis_threshold=vis_threshold,
            return_gt_segmentation=return_gt_segmentation,
            reid_on_gt_model_name=reid_on_gt_model_name,
            only_obj_w_mask=only_obj_w_mask,
            root_dir=original_data_root_dir,
        )

        detection_path = os.path.join(precomputed_seq_dir, "detection.pth")
        self._sequence_detection = torch.load(detection_path)

        self._return_reid_on_det = False
        if not reid_on_det_model_name is None:
            reid_dir = os.path.join(
                precomputed_seq_dir, reid_on_det_model_name, "reid_on_det"
            )
            if os.path.exists(reid_dir):
                self._frame_reid_paths = [
                    os.path.join(reid_dir, frame_reid_file)
                    for frame_reid_file in listdir_nohidden(reid_dir)
                ]
                self._return_reid_on_det = True
                print(self._frame_reid_paths)
            else:
                print(
                    "You requestes reid on detection, but there is no such precompeted data in the given sequence dir!"
                )

        self._return_det_segmentation = False
        if return_det_segmentation:
            segmentation_dir = os.path.join(precomputed_seq_dir, "segmentation")
            if os.path.exists(segmentation_dir):
                self._frame_segmentation_paths = [
                    os.path.join(segmentation_dir, frame_segmentation_file)
                    for frame_segmentation_file in os.listdir(segmentation_dir)
                ]
                self._return_det_segmentation = True
            else:
                print(
                    "You requestes segmentation on detection, but there is no such precompeted data in the given sequence dir!"
                )

    def __getitem__(self, idx):
        # tracking data
        frame_data = self.data[idx]
        frame_detection = self._sequence_detection[idx]
        frame_img = TF.to_tensor(
            Image.open(frame_data["im_path"]).convert("RGB")
        )
        frame = {
            "boxes": frame_detection["boxes"],
            "scores": frame_detection["scores"],
            "gt": frame_data["gt"],
            "vis": frame_data["vis"],
            "img": frame_img,
        }

        # precomputed detection / segmentation
        if self._return_det_segmentation:
            frame_encoded_mask = TF.to_tensor(
                Image.open(self._frame_segmentation_paths[idx])
            ).squeeze(0)
            frame_scalar_mask = decode_segmentation(frame_encoded_mask)
            frame_binary_masks = mask_convert(
                frame_scalar_mask, "scalar", "binary"
            )
            frame["masks"] = frame_binary_masks

        # precomputed reid
        if self._return_reid_on_det:
            reid = torch.load(self._frame_reid_paths[idx])
            frame["reid"] = reid

        return frame
