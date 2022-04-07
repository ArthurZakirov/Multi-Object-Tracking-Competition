import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import random
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from src.detector.object_detector import init_detector_from_config
from src.utils.file_utils import ensure_dir
from src.tracker.data_track import MOT16Sequences
from src.detector.data_obj_detect import MOT16ObjDetect
from src.detector.utils import (
    encode_segmentation,
    mask_convert,
    obj_detect_transforms,
    run_obj_detect,
    convert_frames,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_detect_path",
    type=str,
    default="models/obj_detect/default_maskrcnn.pth",
)
parser.add_argument("--tracker_or_detector_data", type=str, default="tracker")
parser.add_argument("--data_root_dir", type=str, default="data/MOT16")
parser.add_argument(
    "--output_detections_dir", type=str, default="data/precomputed_detection"
)
parser.add_argument("--output_eval_dir", type=str, default="results/detector")
parser.add_argument("--split", type=str, default="mini")
parser.add_argument("--sparse_version", action="store_true")
parser.add_argument("--vis_threshold", type=float, default=0.25)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


def main():
    dataset_name = f"MOT16-{args.split}"
    detector_name = os.path.basename(args.obj_detect_path).split(".")[0]
    # detector_name += "_nms" + str(int(100 * args.nms_thresh))
    file_name = detector_name + "_" + dataset_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj_detect = torch.load(args.obj_detect_path)
    obj_detect.to(device)

    dataset_test = MOT16ObjDetect(
        root=args.data_root_dir,
        split=args.split,
        sparse_version=args.sparse_version,
        transforms=obj_detect_transforms(train=False),
        vis_threshold=args.vis_threshold,
        segmentation=False,
    )

    if args.tracker_or_detector_data == "detector":
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

        print("\ndetector creates predictions...")
        detection_dict = run_obj_detect(
            model=obj_detect, data_loader=data_loader_test, debug=args.debug
        )

        print("\nevaluate detections...")
        eval_dict = dataset_test.evaluate_detections(detection_dict)

        print("\nsave evaluation...")
        output_eval_path = os.path.join(
            args.output_eval_dir, file_name + ".json"
        )
        ensure_dir(output_eval_path)
        with open(output_eval_path, "w") as f:
            json.dump(eval_dict, f)

    elif args.tracker_or_detector_data == "tracker":
        sequences = MOT16Sequences(
            root_dir=args.data_root_dir,
            dataset=dataset_name,
            vis_threshold=args.vis_threshold,
        )

        obj_detect.eval()
        detection_dict = defaultdict(list)
        segmentation_dict = defaultdict(list)
        print("\ndetector creates predictions...")
        for sequence in tqdm(sequences, desc="detect sequence", leave=True):
            for frame_id, frame in tqdm(
                enumerate(sequence, start=1),
                total=len(sequence),
                desc="detect frame",
                leave=False,
            ):
                (image,), (target,) = convert_frames(
                    frames=[frame], inp_fmt="tracker", out_fmt="detector"
                )
                with torch.no_grad():
                    detection = obj_detect([image.to(device)])[0]

                pedestrian = detection["labels"] == 1

                detection_dict[str(sequence)].append(
                    {
                        "boxes": detection["boxes"][pedestrian].cpu(),
                        "scores": detection["scores"][pedestrian].cpu(),
                    }
                )
                if "masks" in detection.keys():
                    binary_masks = (
                        detection["masks"][pedestrian].squeeze(1).int().cpu()
                    )
                    binary_masks = (binary_masks > 0.5).int()
                    scalar_mask = mask_convert(binary_masks, "binary", "scalar")
                    encoded_mask = encode_segmentation(scalar_mask)
                    pil_mask = torchvision.transforms.ToPILImage()(encoded_mask)
                    segmentation_dict[str(sequence)].append(pil_mask)

                    assert not (binary_masks.sum((1, 2)) == 0).any()
                    assert len(torch.unique(scalar_mask)) - 1 == len(
                        detection_dict[str(sequence)][-1]["boxes"]
                    )

                if args.debug and frame_id == 2:
                    break
        print("Done!")

        print("\nevaluate detections...")
        eval_df = dataset_test.evaluate_detections_on_tracking_data(
            detection_dict
        )
        print("Done!")

        print("\nsave detections...")
        for sequence_name, seq_detection in detection_dict.items():
            output_detection_path = os.path.join(
                args.output_detections_dir,
                detector_name,
                sequence_name,
                "detection.pth",
            )
            ensure_dir(output_detection_path)
            torch.save(seq_detection, output_detection_path)
        print("Done!")

        if "masks" in detection.keys():
            print("\nsave segmentation...")
            for sequence_name, frames_masks in segmentation_dict.items():
                for frame_id, pil_mask in enumerate(frames_masks, start=1):
                    output_segmentation_path = os.path.join(
                        args.output_detections_dir,
                        detector_name,
                        sequence_name,
                        "segmentation",
                        f"{frame_id:06d}.png",
                    )
                    ensure_dir(output_segmentation_path)
                    pil_mask.save(output_segmentation_path)
            print("Done!")

        print("\nsave evaluation...")
        output_eval_path = os.path.join(
            args.output_eval_dir, file_name + ".csv"
        )
        ensure_dir(output_eval_path)
        eval_df.to_csv(output_eval_path)
        print("Done!")


if __name__ == "__main__":
    main()
