import pandas as pd
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from matplotlib.patches import Rectangle
import pandas as pd
import json
import os
import torch
import argparse
from src.tracker.tracker import MyTracker
from src.tracker.data_track_precomputed import MOT16SequencePrecomputed
from src.tracker.data_track import load_detection_from_txt
from src.tracker.tracking_analysis import find_switches, find_misses
from src.detector.utils import mask_convert
from src.detector.visualize import colour_map_of_binary_masks


def visualize_frame_fp(
    df, sequence, track, frame_id, show_segmentation=True, figsize=(20, 20)
):
    frame_df = df.groupby("FrameId").get_group(frame_id)
    img = sequence[frame_id]["img"]
    fp_df = frame_df.groupby("Type").get_group("FP")
    fp_hids = fp_df["HId"].astype(int).tolist()
    fp_det_boxes = [track["boxes"][frame_id + 1][hid + 1] for hid in fp_hids]

    fig, fp_ax = plt.subplots(figsize=figsize)
    fp_ax.imshow(img.permute(1, 2, 0))
    fp_ax.axis("off")
    fp_ax.set_title("False Positive")
    for det_box in fp_det_boxes:
        det_box = torchvision.ops.box_convert(det_box, "xyxy", "xywh")
        det_rect = Rectangle(
            det_box[:2], det_box[2], det_box[3], fill=False, color="r"
        )
        fp_ax.add_patch(det_rect)
    fp_ax.plot(0, 0, color="r", label="false positive")
    fp_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if show_segmentation:
        masks = sequence[frame_id]["masks"]
        colour_map = colour_map_of_binary_masks(masks)
        fp_ax.imshow(colour_map)
    return fig


def visualize_frame_misses(
    df, sequence, track, frame_id, show_segmentation=False, figsize=(20, 20)
):
    frame_df = df.groupby("FrameId").get_group(frame_id)
    img = sequence[frame_id]["img"]
    miss_df = frame_df.groupby("Type").get_group("MISS")
    miss_oids = miss_df["OId"].astype(int).tolist()
    assert len(miss_oids) > 0
    miss_gt_boxes = [sequence[frame_id]["gt"][oid] for oid in miss_oids]

    fig, fn_ax = plt.subplots(figsize=figsize)
    fn_ax.imshow(img.permute(1, 2, 0))
    fn_ax.axis("off")
    fn_ax.set_title("False Negative")
    for gt_box in miss_gt_boxes:
        gt_box = torchvision.ops.box_convert(gt_box, "xyxy", "xywh")
        gt_rect = Rectangle(
            gt_box[:2], gt_box[2], gt_box[3], fill=False, color="b"
        )
        fn_ax.add_patch(gt_rect)
    fn_ax.plot(0, 0, color="b", label="false negative")
    fn_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return fig


def visualize_frame_matches(
    df, sequence, track, frame_id, show_segmentation=True, figsize=(20, 20)
):
    frame_df = df.groupby("FrameId").get_group(frame_id)
    img = sequence[frame_id]["img"]
    match_df = frame_df.groupby("Type").get_group("MATCH")
    match_oids = match_df["OId"].astype(int).tolist()
    assert len(match_oids) > 0
    match_hids = match_df["HId"].astype(int).tolist()
    matched_det_boxes = [
        track["boxes"][frame_id + 1][hid + 1] for hid in match_hids
    ]
    matched_gt_boxes = [sequence[frame_id]["gt"][oid] for oid in match_oids]

    fig, tp_ax = plt.subplots(figsize=figsize)
    tp_ax.imshow(img.permute(1, 2, 0))
    tp_ax.axis("off")
    tp_ax.set_title("True Positive")
    for det_box, gt_box in zip(matched_det_boxes, matched_gt_boxes):
        gt_box = torchvision.ops.box_convert(gt_box, "xyxy", "xywh")
        det_box = torchvision.ops.box_convert(det_box, "xyxy", "xywh")
        gt_rect = Rectangle(
            gt_box[:2], gt_box[2], gt_box[3], fill=False, color="b"
        )
        det_rect = Rectangle(
            det_box[:2], det_box[2], det_box[3], fill=False, color="g"
        )
        tp_ax.add_patch(gt_rect)
        tp_ax.add_patch(det_rect)
    tp_ax.plot(0, 0, color="g", label="true positive")
    tp_ax.plot(0, 0, color="b", label="ground truth")
    tp_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if show_segmentation:
        masks = sequence[frame_id]["masks"]
        colour_map = colour_map_of_binary_masks(masks)
        tp_ax.imshow(colour_map)
    return fig


def visualize_frame_detections(
    df,
    sequence,
    track,
    frame_id,
    type,
    show_segmentation=True,
    figsize=(20, 20),
):
    """
    Arguments
    ---------
    df 
        - pandas.Dataframe from mot_accum.mot_events 

    sequence
        - MOT16SequencePrecomputed object

    track
        - dict of tracks in gt format: track[frame_id][hypothesis_id] = box (xyxy)
        - note that frame_id and hypothesis_id start at 1

    frame_id
        - start from 1

    type 
        - "MATCH", "MISS", "FP"
    """

    if type == "FP":
        fig = visualize_frame_fp(
            df, sequence, track, frame_id, show_segmentation, figsize
        )

    if type == "MISS":
        fig = visualize_frame_misses(
            df, sequence, track, frame_id, show_segmentation, figsize
        )

    if type == "MATCH":
        fig = visualize_frame_matches(
            df, sequence, track, frame_id, show_segmentation, figsize
        )
    return fig


def visualize_switches(
    df, sequence, track, show_idx=0, zoom=True, figsize=(15, 15)
):
    """
    Arguments
    ---------
    df 
        - pandas.Dataframe from mot_accum.mot_events 

    sequence
        - MOT16SequencePrecomputed object

    track
        - dict of tracks in gt format: track[frame_id][hypothesis_id] = box (xyxy)
        - note that frame_id and hypothesis_id start at 1

    show_idx 
        - list of indices of switch events
        - during first functions call there will be print statement that tells which switches exist
    """
    (
        switch_idxs,
        last_match_idxs,
        ascend_bool,
        active_switch_bool,
    ) = find_switches(df)
    print(
        f"There are {len(switch_idxs)} switches in the sequence, pick a 'show_switch_idx' between [0 - {len(switch_idxs)-1}]."
    )

    fig, ax = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=False, squeeze=False,
    )

    switch_idx = switch_idxs[show_idx]
    match_idx = last_match_idxs[show_idx]

    switch_frame_id = df.loc[switch_idx, "FrameId"]
    match_frame_id = df.loc[match_idx, "FrameId"]
    switch_hid = int(df.loc[switch_idx, "HId"])
    match_hid = int(df.loc[match_idx, "HId"])
    switch_img = sequence[switch_frame_id]["img"]
    match_img = sequence[match_frame_id]["img"]

    switch_box = track["boxes"][switch_frame_id + 1][switch_hid + 1]
    switch_rect = torchvision.ops.box_convert(switch_box, "xyxy", "xywh")
    match_box = track["boxes"][match_frame_id + 1][match_hid + 1]
    match_rect = torchvision.ops.box_convert(match_box, "xyxy", "xywh")

    ax[0, 0].imshow(match_img.permute(1, 2, 0))
    ax[0, 0].add_patch(
        Rectangle(
            match_rect[:2],
            match_rect[2],
            match_rect[3],
            fill=False,
            color="g",
            label="last matched detection",
        )
    )

    ax[0, 1].imshow(switch_img.permute(1, 2, 0))
    ax[0, 1].add_patch(
        Rectangle(
            switch_rect[:2],
            switch_rect[2],
            switch_rect[3],
            fill=False,
            color="r",
            label="new detection",
        )
    )
    ax[0, 1].add_patch(
        Rectangle(
            match_rect[:2],
            match_rect[2],
            match_rect[3],
            fill=False,
            color="g",
            label="last matched detection",
        )
    )

    ax[0, 0].tick_params(axis="both", labelsize=0, length=0)
    ax[0, 0].set_ylabel(f"frame_id: {match_frame_id}")

    ax[0, 1].tick_params(axis="both", labelsize=0, length=0)
    ax[0, 1].set_ylabel(f"frame_id: {switch_frame_id}")

    ax[0, 0].set_title("Last Detection before ID-Loss", fontsize=20)
    ax[0, 1].set_title("First Detection with new ID", fontsize=20)

    ax[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    ax[0, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    heading = "Note:\n"
    ascend_txt = "The switch is an ascend. This means that the object got assigned a new track ID."
    swap_txt = "The switch is an swap. This means that the object got assigned to a different, previously existing track ID."
    txt = heading + ascend_txt if ascend_bool[show_idx] else swap_txt

    fig.text(0.5, 1, txt, ha="center", va="top")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    if zoom:
        ax[0, 0].axis(
            [
                match_box[0] - 100,
                match_box[2] + 100,
                match_box[3] + 100,
                match_box[1] - 100,
            ]
        )
        ax[0, 1].axis(
            [
                switch_box[0] - 100,
                switch_box[2] + 100,
                switch_box[3] + 100,
                switch_box[1] - 100,
            ]
        )
    return fig


def visualize_lost_tracks(
    events_df,
    sequence,
    track,
    tracker,
    show_idx=0,
    show_boxes=["track", "fut", "pred"],
    zoom=True,
    figsize=(15, 15),
):
    miss_sequences, last_match_sequences = find_misses(events_df)
    show_idxs = [
        idx
        for (idx, last_match_sequenc) in enumerate(last_match_sequences)
        if last_match_sequenc
    ]
    miss_sequences = [miss_sequences[idx] for idx in show_idxs]
    last_match_sequences = [last_match_sequences[idx] for idx in show_idxs]

    print("\n\n\nInformation\n-------------------")
    print(
        f"There are {len(show_idxs)} false negatives due to track losses in this scene."
    )
    print(
        f"Select an event [0 - {len(show_idxs)-1}] that you want to visualize."
    )

    miss_df = (
        events_df.loc[miss_sequences[show_idx]].copy().reset_index(drop=True)
    )
    last_match_df = (
        events_df.loc[last_match_sequences[show_idx]]
        .copy()
        .reset_index(drop=True)
    )

    oid = last_match_df.loc[0, "OId"]
    hid = last_match_df.loc[0, "HId"]
    frame_id = int(last_match_df.loc[len(last_match_df) - 1, "FrameId"])

    gt_hist_traj = torch.stack(
        [
            sequence[frame_id]["gt"][oid]
            for frame_id in last_match_df["FrameId"].tolist()
        ],
        dim=0,
    )

    det_hist_traj = torch.stack(
        [
            track["boxes"][frame_id + 1][hid + 1]
            for frame_id in last_match_df["FrameId"].tolist()
        ],
        dim=0,
    )
    fut_traj = torch.stack(
        [sequence[frame_id]["gt"][oid] for frame_id in miss_df["FrameId"]]
    )

    det_hist_traj = tracker.short_motion_predictor.kalman.smooth(det_hist_traj)
    tracker.short_motion_predictor.kalman.reset_state()
    pred_traj = tracker.short_motion_predictor(
        [det_hist_traj], future_lens=[len(fut_traj)]
    )[0]

    gt_hist_plot_traj = torch.stack(
        [0.5 * (gt_hist_traj[:, 0] + gt_hist_traj[:, 2]), gt_hist_traj[:, 3]],
        dim=1,
    )
    det_hist_plot_traj = torch.stack(
        [
            0.5 * (det_hist_traj[:, 0] + det_hist_traj[:, 2]),
            det_hist_traj[:, 3],
        ],
        dim=1,
    )
    fut_plot_traj = torch.stack(
        [0.5 * (fut_traj[:, 0] + fut_traj[:, 2]), fut_traj[:, 3]], dim=1
    )
    fut_plot_traj = torch.cat([gt_hist_plot_traj[[-1]], fut_plot_traj], dim=0)
    pred_plot_traj = torch.stack(
        [0.5 * (pred_traj[:, 0] + pred_traj[:, 2]), pred_traj[:, 3]], dim=1
    )
    pred_plot_traj = torch.cat(
        [det_hist_plot_traj[[-1]], pred_plot_traj], dim=0
    )

    gt_rect = torchvision.ops.box_convert(gt_hist_traj[-1], "xyxy", "xywh")
    det_rect = torchvision.ops.box_convert(det_hist_traj[-1], "xyxy", "xywh")
    fut_rect = torchvision.ops.box_convert(fut_traj[0], "xyxy", "xywh")
    pred_rect = torchvision.ops.box_convert(pred_traj[-1], "xyxy", "xywh")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(sequence[frame_id]["img"].permute(1, 2, 0))

    if "track" in show_boxes:
        ax.plot(det_hist_plot_traj[:, 0], det_hist_plot_traj[:, 1], color="y")
        ax.add_patch(
            Rectangle(
                det_rect[:2],
                det_rect[2],
                det_rect[3],
                fill=False,
                color="y",
                label="tracked history before miss",
            )
        )

    if "pred" in show_boxes:
        ax.plot(pred_plot_traj[:, 0], pred_plot_traj[:, 1], color="r")
        ax.scatter(pred_plot_traj[0, 0], pred_plot_traj[0, 1], color="r")
        ax.add_patch(
            Rectangle(
                pred_rect[:2],
                pred_rect[2],
                pred_rect[3],
                fill=False,
                color="r",
                label="prediction after miss",
            )
        )

    if "fut" in show_boxes:
        ax.plot(fut_plot_traj[:, 0], fut_plot_traj[:, 1], color="g")
        ax.scatter(fut_plot_traj[0, 0], fut_plot_traj[0, 1], color="g")
        ax.add_patch(
            Rectangle(
                fut_rect[:2],
                fut_rect[2],
                fut_rect[3],
                fill=False,
                color="g",
                label="ground truth after miss",
            )
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    ax.axis("off")

    fig.text(
        0.02,
        0.5,
        f"\nPast detected length: {len(det_hist_traj)} frames\nFuture undetected length: {len(fut_traj)} frames",
        transform=plt.gcf().transFigure,
    )

    if zoom:
        ax.axis(
            [
                det_hist_traj[-1, 0] - 100,
                det_hist_traj[-1, 2] + 100,
                det_hist_traj[-1, 3] + 100,
                det_hist_traj[-1, 1] - 100,
            ]
        )
    return fig


def get_visualization_functions(sequence_dir):
    """
    Arguments
    ---------
    sequence_dir
        - path to tracker results directory of a certain sequence
        - p.e. "results\\tracker\\07-04-2022_18-33\\MOT16-02"
        - make sure the dir contains the following files
            - "track.txt"
            - "events.csv"
            - "eval_config.json"

    Returns
    -------
    lambda functions that only require reduced input to plot results
    """
    track_path = os.path.join(sequence_dir, "track.txt")
    track = load_detection_from_txt(track_path, vis_threshold=0.0, mode="track")

    events_path = os.path.join(sequence_dir, "events.csv")
    events_df = pd.read_csv(events_path)

    eval_config_path = os.path.join(
        os.path.dirname(sequence_dir), "eval_config.json"
    )
    with open(eval_config_path, "r") as f:
        eval_config = json.load(f)
    args = argparse.Namespace(**eval_config)

    with open(args.tracker_config_path, "r") as f:
        tracker_hyperparams = json.load(f)

    tracker = MyTracker.from_config(tracker_hyperparams)

    seq_name = os.path.basename(sequence_dir)
    sequence = MOT16SequencePrecomputed(
        seq_name=seq_name,
        original_data_root_dir=args.original_data_root_dir,
        precomputed_seq_dir=os.path.join(
            args.precomputed_data_root_dir, seq_name
        ),
        vis_threshold=args.vis_threshold,
        return_det_segmentation=tracker.assign_model.use_segmentation,
        return_gt_segmentation=tracker.assign_model.use_segmentation,
    )

    visualize_functions = {
        "detections": lambda frame_id, type: visualize_frame_detections(
            events_df, sequence, track, frame_id, type
        ),
        "switches": lambda show_idx: visualize_switches(
            events_df, sequence, track, show_idx
        ),
        "lost_tracks": lambda lost_idx, show_boxes: visualize_lost_tracks(
            events_df, sequence, track, tracker, lost_idx, show_boxes
        ),
    }
    return visualize_functions

