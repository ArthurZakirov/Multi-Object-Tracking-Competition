import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torchvision

from src.tracker.data_track import load_detection_from_txt
from src.tracker.data_track import split_sequence_names
from src.motion_prediction.kalman import (
    KalmanFilter,
    freeze_at_mean,
    obj_is_moving,
)


def sliding_windows(sequence, history_len=10, future_len=5):
    hist_crops = []
    fut_crops = []
    for step in range(history_len, len(sequence) - future_len):
        hist_crops.append(sequence[step - history_len : step])
        fut_crops.append(sequence[step : step + future_len])
    return hist_crops, fut_crops


class MOT16MotionPrediction(torch.utils.data.Dataset):
    def __init__(
        self, root, split, use_filter=True, future_len=20, history_len=20
    ):
        super().__init__()
        self._hist_trajs = []
        self._fut_trajs = []
        self._hist_trajs_processed = []
        self._fut_trajs_processed = []
        self._train_folders = os.listdir(os.path.join(root, "train"))
        self._test_folders = os.listdir(os.path.join(root, "test"))

        kalman = KalmanFilter(
            process_variance=500, measurement_variance=0.001, dt=1 / 30
        )

        seq_names = split_sequence_names(split, root)
        for seq_name in seq_names:
            if seq_name == "MOT16-05":
                continue
            if seq_name in self._train_folders:
                gt_path = os.path.join(root, "train", seq_name, "gt", "gt.txt")
            else:
                gt_path = os.path.join(root, "test", seq_name, "gt", "gt.txt")
            gt = load_detection_from_txt(gt_path)

            boxes_xyxy = defaultdict(list)

            for (frame_id, frame_dict) in gt["boxes"].items():
                for (obj_id, box) in frame_dict.items():
                    boxes_xyxy[obj_id].append(box)

            for obj_id, obj_boxes in boxes_xyxy.items():
                full_traj = torch.stack(obj_boxes, dim=0)
                full_traj = torchvision.ops.box_convert(
                    full_traj, "xyxy", "cxcywh"
                )
                hist, fut = sliding_windows(
                    full_traj, future_len=future_len, history_len=history_len
                )
                self._hist_trajs.extend(hist)
                self._fut_trajs.extend(fut)

                kalman_traj = full_traj.clone()
                kalman_traj = kalman.smooth(full_traj)
                kalman.reset_state()
                hist_filt, fut_filt = sliding_windows(
                    kalman_traj, future_len=future_len, history_len=history_len
                )
                self._hist_trajs_processed.extend(hist_filt)
                self._fut_trajs_processed.extend(fut_filt)

        self._hist_trajs_processed = [
            freeze_at_mean(traj) if not obj_is_moving(traj) else traj
            for traj in self._hist_trajs_processed
        ]
        self._fut_trajs_processed = [
            freeze_at_mean(traj) if not obj_is_moving(traj) else traj
            for traj in self._fut_trajs_processed
        ]

    def __len__(self):
        return len(self._hist_trajs)

    def __getitem__(self, idx, processed=True):
        if processed:
            x = self._hist_trajs_processed[idx]
            y = self._fut_trajs_processed[idx]
        else:
            x = self._hist_trajs[idx]
            y = self._fut_trajs[idx]
        return x, y
