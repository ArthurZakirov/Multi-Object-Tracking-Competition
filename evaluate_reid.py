import os
import random
import argparse
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from src.tracker.data_track_precomputed import MOT16SequencesPrecomputed
from src.tracker.tracker import MyTracker
from src.tracker.data_track import MOT16Sequences
from src.utils.file_utils import ensure_dir
from src.tracker.tracker import MyTracker
from src.detector.utils import convert_frames

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--reid_model_dir",
    type=str,
    default="models/reid_model/default_reid",
    help="path to tracker configuration",
)

parser.add_argument(
    "--original_data_root_dir",
    type=str,
    default="data/MOT16",
    help="path to evaluation data root",
)
parser.add_argument(
    "--precomputed_data_root_dir",
    type=str,
    default="data/precomputed_detection/coco_maskrcnn_recall",
    help="path to precomputed evaluation data root",
)
parser.add_argument(
    "--split",
    type=str,
    default="mini",
    help="part of dataset, choose from ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', 'reid', 'train_wo_val', 'train_wo_val2', 'val', 'val2']",
)

parser.add_argument(
    "--vis_threshold",
    type=float,
    default=0.0,
    help="Threshold of visibility of persons above which they are selected",
)


args = parser.parse_args()


def main():
    reid_model_path = os.path.join(args.reid_model_dir, "model.pth")
    reid_model = torch.load(reid_model_path)
    reid_model_name = os.path.basename(args.reid_model_dir)

    # TODO : this is a temporary fix, because input_is_masked is not included as an attribute in
    # reid_model.input_is_masked = json.load(
    #     open(os.path.join(args.reid_model_dir, "model_config.json"), "r")
    # )["input_is_masked"]

    reid_model_name = "masked_reid"
    reid_model.input_is_masked = True

    # execute
    sequences = MOT16SequencesPrecomputed(
        precomputed_data_root_dir=args.precomputed_data_root_dir,
        original_data_root_dir=args.original_data_root_dir,
        split=args.split,
        vis_threshold=args.vis_threshold,
        return_det_segmentation=reid_model.input_is_masked,
        return_gt_segmentation=reid_model.input_is_masked,
    )

    tracker = MyTracker(reid_model=reid_model)

    print("\nrun_reid...")
    for sequence in tqdm(sequences, desc="sequences", leave=True):
        for frame_id, frame in tqdm(
            enumerate(sequence, start=1),
            total=len(sequence),
            desc="frame",
            leave=False,
        ):

            det_boxes, _, det_masks = tracker._get_detection(frame)
            det_features = tracker._get_reid_features(
                frame, det_boxes, det_masks
            )
            output_reid_on_det_path = os.path.join(
                args.precomputed_data_root_dir,
                str(sequence),
                reid_model_name,
                "reid_on_det",
                f"{frame_id:06d}.pth",
            )
            ensure_dir(output_reid_on_det_path)
            torch.save(det_features, output_reid_on_det_path)

            _, (targets,) = convert_frames([frame], "tracker", "detector")
            if reid_model.input_is_masked:
                gt_features = tracker._get_reid_features(
                    frame, boxes=targets["boxes"], masks=targets["seg_img"]
                )
            else:
                gt_features = tracker._get_reid_features(
                    frame, boxes=targets["boxes"]
                )
            output_reid_on_gt_path = os.path.join(
                args.precomputed_data_root_dir,
                str(sequence),
                reid_model_name,
                "reid_on_gt",
                f"{frame_id:06d}.pth",
            )
            ensure_dir(output_reid_on_gt_path)
            torch.save(gt_features, output_reid_on_gt_path)


if __name__ == "__main__":
    main()
