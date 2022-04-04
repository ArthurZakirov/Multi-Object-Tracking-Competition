import os
import argparse
from datetime import datetime
from collections import defaultdict
import random
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.file_utils import ensure_dir
from src.tracker.data_track import MOT16Sequences
from src.detector.data_obj_detect import MOT16ObjDetect
from src.detector.utils import (
    obj_detect_transforms,
    evaluate_obj_detect,
    convert_frames,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_detect_config_path",
    type=str,
    default="config/obj_detect/maskrcnn.pth",
)
parser.add_argument("--tracker_or_detector_data", type=str, default="tracker")
parser.add_argument("--data_root_dir", type=str, default="data/MOT16")
parser.add_argument(
    "--output_detections_dir", type=str, default="data/precomputed_detection"
)
parser.add_argument("--output_eval_dir", type=str, default="results/detector")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--vis_threshold", type=float, default=0.25)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


def main():
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    dataset_name = f"MOT16-{args.split}"
    detector_name = os.path.basename(args.obj_detect_path).split(".")[0]
    file_name = detector_name + "_" + dataset_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj_detect = torch.load(args.obj_detect_path)
    obj_detect.to(device)

    dataset_test = MOT16ObjDetect(
        root=os.path.join(args.data_root_dir, "train"),
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
        detection_dict = evaluate_obj_detect(
            model=obj_detect, data_loader=data_loader_test, debug=True
        )

        print("\nevaluate detections...")
        eval_dict = dataset_test.evaluate_detections(detection_dict)

    elif args.tracker_or_detector_data == "tracker":
        sequences = MOT16Sequences(
            root_dir=args.data_root_dir,
            dataset=dataset_name,
            vis_threshold=args.vis_threshold,
            segmentation=False,
        )

        obj_detect.eval()
        detection_dict = defaultdict(list)
        print("\ndetector creates predictions...")
        for sequence in tqdm(sequences, desc="detect sequence", leave=True):
            for frame in tqdm(sequence, desc="detect frame", leave=False):
                (image,), (target,) = convert_frames(
                    [frame], "tracker", "detector"
                )
                with torch.no_grad():
                    pred = obj_detect([image.to(device)])[0]

                detection_dict[str(sequence)].append(
                    {
                        "det": {
                            "boxes": pred["boxes"].cpu(),
                            "scores": pred["scores"].cpu(),
                        },
                        "gt": target,
                    }
                )
                if args.debug:
                    break
            if args.debug:
                break
            print("Done!")

        print("\nevaluate detections...")
        eval_df = dataset_test.evaluate_detections_on_tracking_data(
            detection_dict
        )
        print("Done!")

        print("\nsave detections...")
        output_detection_path = os.path.join(
            args.output_detections_dir, file_name + ".pth"
        )
        ensure_dir(output_detection_path)
        torch.save(detection_dict, output_detection_path)
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
