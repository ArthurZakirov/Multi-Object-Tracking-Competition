from ast import arg
import matplotlib.pyplot as plt
import os
import random
import argparse
from datetime import datetime
import json
from tqdm import tqdm

import torch
import numpy as np
from src.tracker.data_track_precomputed import MOT16SequencesPrecomputed
from src.tracker.tracker import MyTracker
from src.tracker.utils import run_tracker, write_results, get_mot_accum
from src.tracker.data_track import MOT16Sequences
from src.utils.file_utils import ensure_dir
import motmetrics as mm

with open("results\\tracker\\06-04-2022_23-04\\eval_config.json", "r") as f:
    eval_config = json.load(f)
args = argparse.Namespace(**eval_config)

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
        return_det_segmentation=tracker.assign_model.use_segmentation,
        return_gt_segmentation=tracker.assign_model.use_segmentation,
    )

sequence = sequences[0]


with torch.no_grad():
    for frame in tqdm(sequence, desc="frame", leave=False):
        tracker.step(frame)

mot_accum = get_mot_accum(tracker.get_results(), sequence)
events_df = mot_accum.mot_events
events_df.to_csv("events.csv")
