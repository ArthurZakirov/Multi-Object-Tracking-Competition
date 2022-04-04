import os
import random
import argparse
from datetime import datetime
import json
from this import d

import torch
import numpy as np
from src.tracker.data_track_precomputed import MOT16SequencesPrecomputed
from src.tracker.tracker import MyTracker
from src.tracker.utils import run_tracker, write_results
from src.tracker.data_track import MOT16Sequences
from src.utils.file_utils import ensure_dir

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--original_data_root_dir",
    type=str,
    default="data/MOT16",
    help="path to evaluation data root",
)
parser.add_argument(
    "--precomputed_data_root_dir",
    type=str,
    default="data/precomputed_detection/maskrcnn",
    help="path to precomputed evaluation data root",
)
parser.add_argument("--use_precomputed", action="store_true")
parser.add_argument(
    "--split",
    type=str,
    default="train",
    help="part of dataset, choose from ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', 'reid', 'train_wo_val', 'train_wo_val2', 'val', 'val2']",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="results/tracker",
    help="path to results dir",
)

parser.add_argument(
    "--tracker_config_path",
    type=str,
    default="config/tracker/tracker.json",
    help="path to tracker configuration",
)


parser.add_argument(
    "--vis_threshold",
    type=float,
    default=0.0,
    help="Threshold of visibility of persons above which they are selected",
)

parser.add_argument("--save_evaluation", action="store_true")

parser.add_argument("--save_eval_config", action="store_true")

parser.add_argument("--save_tracker_predictions", action="store_true")

args = parser.parse_args()


def main():
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    dataset = f"MOT16-{args.split}"
    with open(args.tracker_config_path, "r") as f:
        tracker_hyperparams = json.load(f)

    # execute
    tracker = MyTracker.from_config(tracker_hyperparams)

    if args.use_precomputed:
        sequences = MOT16SequencesPrecomputed(
            precomputed_data_root_dir=args.precomputed_data_root_dir,
            original_data_root_dir=args.original_data_root_dir,
            split=args.split,
            vis_threshold=args.vis_threshold,
        )

    else:
        sequences = MOT16Sequences(
            dataset=dataset,
            root_dir=args.data_root_dir,
            vis_threshold=args.vis_threshold,
        )

    print("\nrun_tracker...")
    eval_df, results_seq = run_tracker(sequences=sequences, tracker=tracker)

    if args.save_eval_config:
        print("\nsave_eval_config...")
        output_eval_config_path = os.path.join(
            args.output_dir, time, "eval_config.json"
        )
        ensure_dir(output_eval_config_path)
        eval_config = {
            "split": args.split,
            "tracker_config_path": args.tracker_config_path,
            "precomputed_detection_path": args.precomputed_detection_path,
            "vis_threshold": args.vis_threshold,
        }
        with open(output_eval_config_path, "w") as f:
            json.dump(eval_config)

    if args.save_evaluation:
        print("\nsave_evaluation...")
        output_evaluation_path = os.path.join(
            args.output_dir, time, "tracker_evaluation_{dataset}.csv"
        )
        ensure_dir(output_evaluation_path)
        if not eval_df is None:
            eval_df.to_csv(output_evaluation_path)

    if args.save_tracker_predictions:
        print("\nsave_tracker_predictions...")
        output_predictions_dir = os.path.join(
            args.output_dir, time, "tracker_predictions_{dataset}"
        )
        for sequence_name, tracker_results_dict in results_seq.items():
            output_path = os.path.join(
                output_predictions_dir, sequence_name + ".txt"
            )
            ensure_dir(output_path)
            write_results(tracker_results_dict, output_path)


if __name__ == "__main__":
    main()
