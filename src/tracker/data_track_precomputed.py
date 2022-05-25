from msilib import sequence
import os
import torch
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from src.detector.data_utils import decode_segmentation, load_segmentation
from src.utils.file_utils import listdir_nohidden
from src.tracker.data_track import MOT16, MOT16Sequence, split_sequence_names
from src.tracker.utils import rgb2gray, get_crops_from_boxes

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

    def get_detection_results_dict(self):
        from collections import defaultdict

        results = defaultdict(list)
        for sequence in self._sequences:
            for frame_id in range(len(sequence)):
                frame_result = sequence._sequence_detection[frame_id]
                results[str(sequence)].append(frame_result)
        return results


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
        return_image=True,
        return_statistical_info=False,
    ):
        super().__init__(
            seq_name=seq_name,
            vis_threshold=vis_threshold,
            return_gt_segmentation=return_gt_segmentation,
            reid_on_gt_model_name=reid_on_gt_model_name,
            only_obj_w_mask=only_obj_w_mask,
            root_dir=original_data_root_dir,
        )

        self._return_image = return_image
        self._return_statistical_info = return_statistical_info

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
        frame = super().__getitem__(idx)
        frame.update(self._sequence_detection[idx])

        # precomputed detection / segmentation
        if self._return_det_segmentation:
            masks, _ = load_segmentation(
                seg_path=self._frame_segmentation_paths[idx]
            )
            frame["masks"] = masks

        # precomputed reid
        if self._return_reid_on_det:
            reid = torch.load(self._frame_reid_paths[idx])
            frame["reid"] = reid

        # darkness: additional stuff that I use for my analysis
        if self._return_statistical_info:
            crops = {
                id: get_crops_from_boxes(
                    boxes=box.unsqueeze(0), image=frame["img"]
                )[0]
                for (id, box) in frame["gt"].items()
            }
            grey_crops = {id: rgb2gray(crop) for (id, crop) in crops.items()}
            luminosity = {id: crop.mean() for (id, crop) in grey_crops.items()}
            contrast = {id: crop.std() for (id, crop) in grey_crops.items()}
            frame["luminosity"] = luminosity
            frame["contrast"] = contrast

            # area: additional stuff that I use for my analysis
            area = {
                id: torchvision.ops.box_area(box.unsqueeze(0)).item()
                for (id, box) in frame["gt"].items()
            }
            frame["area"] = area
        return frame
