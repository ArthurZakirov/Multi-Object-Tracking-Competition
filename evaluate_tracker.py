import os
import random
import argparse
from datetime import datetime
import json
import pandas as pd

import torch
import numpy as np
from src.tracker.data_track_precomputed import MOT16SequencesPrecomputed
from src.tracker.tracker import MyTracker
from src.tracker.utils import run_tracker, write_results, evaluate_mot_accums
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
    default="data/precomputed_detection/coco_maskrcnn_recall",
    help="path to precomputed evaluation data root",
)
parser.add_argument(
    "--reid_on_det_model_name", type=str, default="default_reid",
)

parser.add_argument("--use_precomputed", action="store_true")
parser.add_argument(
    "--split",
    type=str,
    default="mini",
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


args = parser.parse_args()


def main():
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    dataset = f"MOT16-{args.split}"

    with open(args.tracker_config_path, "r") as f:
        tracker_hyperparams = json.load(f)

    print("\nsave_eval_config...")
    output_eval_config_path = os.path.join(
        args.output_dir, time, "eval_config.json"
    )
    ensure_dir(output_eval_config_path)
    with open(output_eval_config_path, "w") as f:
        json.dump(vars(args), f)

    # execute
    from src.tracker.pedestrian_tracker import (
        Tracker,
        KalmanTracker,
        PatientTracker,
        ByteTracker,
        DistractorAwareTracker,
    )

    tracker = DistractorAwareTracker.from_config(tracker_hyperparams)
    # tracker = MyTracker.from_config(tracker_hyperparams)

    if args.use_precomputed:
        sequences = MOT16SequencesPrecomputed(
            precomputed_data_root_dir=args.precomputed_data_root_dir,
            original_data_root_dir=args.original_data_root_dir,
            split=args.split,
            vis_threshold=args.vis_threshold,
            reid_on_det_model_name=args.reid_on_det_model_name,
            return_det_segmentation=tracker.assign_model.use_segmentation,
            return_gt_segmentation=tracker.assign_model.use_segmentation,
        )

    else:
        sequences = MOT16Sequences(
            dataset=dataset,
            root_dir=args.original_data_root_dir,
            vis_threshold=args.vis_threshold,
        )

    print("\nrun_tracker...")
    results_seq, mot_accums = run_tracker(sequences=sequences, tracker=tracker)
    seq_names_with_gt = [
        str(sequence) for sequence in sequences if not sequence.no_gt
    ]

    print("\nevaluate mot accums...")
    eval_df = evaluate_mot_accums(
        accums=mot_accums.copy(),
        names=seq_names_with_gt.copy(),
        generate_overall=True,
    )

    print("\nsave_eval_config...")
    output_eval_config_path = os.path.join(
        args.output_dir, time, "eval_config.json"
    )
    ensure_dir(output_eval_config_path)
    with open(output_eval_config_path, "w") as f:
        json.dump(vars(args), f)

    print("\nsave_evaluation...")
    output_evaluation_path = os.path.join(
        args.output_dir, time, f"tracker_evaluation_{dataset}.csv"
    )
    ensure_dir(output_evaluation_path)
    if not eval_df is None:
        eval_df = eval_df.applymap(
            lambda x: x if isinstance(x, str) else f"{x:.2f}"
        )
        eval_df.to_csv(output_evaluation_path)

    print("\nsave_tracker_predictions...")
    for sequence_name, tracker_results_dict in results_seq.items():
        output_pred_path = os.path.join(
            args.output_dir, time, sequence_name, "track.txt"
        )
        ensure_dir(output_pred_path)
        write_results(tracker_results_dict, output_pred_path)

    for sequence_name, mot_accum in zip(seq_names_with_gt, mot_accums):
        output_events_path = os.path.join(
            args.output_dir, time, sequence_name, "events.csv"
        )
        ensure_dir(output_events_path)
        event_df = mot_accum.mot_events
        event_df.to_csv(output_events_path)


if __name__ == "__main__":
    main()
