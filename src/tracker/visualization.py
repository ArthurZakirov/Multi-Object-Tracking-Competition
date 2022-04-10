import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from matplotlib.patches import Rectangle
import pandas as pd
import json
import os
import argparse
from src.tracker.tracker import MyTracker
from src.tracker.data_track_precomputed import MOT16SequencePrecomputed
from src.tracker.data_track import load_detection_from_txt


def visualize_frame_detections(
    df, sequence, track, frame_id, type="MATCH", figsize=(20, 20)
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
    frame_df = df.groupby("FrameId").get_group(frame_id)
    img = sequence[frame_id]["img"]

    # False positives
    if type == "FP":
        fig, fp_ax = plt.subplots(figsize=figsize)
        fp_df = frame_df.groupby("Type").get_group("FP")
        fp_hids = fp_df["HId"].astype(int).tolist()
        fp_det_boxes = [
            track["boxes"][frame_id + 1][hid + 1] for hid in fp_hids
        ]
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

    # misses
    if type == "MISS":
        fig, fn_ax = plt.subplots(figsize=figsize)
        miss_df = frame_df.groupby("Type").get_group("MISS")
        miss_oids = miss_df["OId"].astype(int).tolist()
        assert len(miss_oids) > 0
        miss_gt_boxes = [sequence[frame_id]["gt"][oid] for oid in miss_oids]
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

    # matches
    if type == "MATCH":
        fig, tp_ax = plt.subplots(figsize=figsize)
        match_df = frame_df.groupby("Type").get_group("MATCH")
        match_oids = match_df["OId"].astype(int).tolist()
        assert len(match_oids) > 0
        match_hids = match_df["HId"].astype(int).tolist()
        matched_det_boxes = [
            track["boxes"][frame_id + 1][hid + 1] for hid in match_hids
        ]
        matched_gt_boxes = [sequence[frame_id]["gt"][oid] for oid in match_oids]
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


def visualize_sequence_switches(df, sequence, track, show_switch_idx=None):
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

    show_switch_idx 
        - list of indices of switch events
        - during first functions call there will be print statement that tells which switches exist
    """
    switch_df = df.groupby("Type").get_group("SWITCH")
    print(f"There are {len(switch_df)} switches in the sequences.")
    switch_oids = switch_df["OId"].unique()
    switch_events = switch_df["Event"].unique()
    switch_idxs = []
    last_match_idxs = []
    for id in switch_oids:
        obj_df = df.groupby("OId").get_group(id)
        obj_switch_idxs = (
            obj_df.groupby("Type").get_group("SWITCH").index.tolist()
        )
        for switch_idx in obj_switch_idxs:
            before_switch_df = obj_df[
                obj_df["FrameId"] < obj_df.loc[switch_idx, "FrameId"]
            ]
            last_match_before_switch_idx = (
                before_switch_df.groupby("Type")
                .get_group("MATCH")
                .index[-1]
                .item()
            )
            last_match_idxs.append(last_match_before_switch_idx)
            switch_idxs.append(switch_idx)

    if not show_switch_idx is None:
        switch_events = [switch_events[idx] for idx in show_switch_idx]
        switch_idxs = [switch_idxs[idx] for idx in show_switch_idx]
        last_match_idxs = [last_match_idxs[idx] for idx in show_switch_idx]

    fig, ax = plt.subplots(
        len(switch_events),
        2,
        figsize=(20, len(switch_events) * 6),
        constrained_layout=False,
        squeeze=False,
    )
    for event, (switch_idx, match_idx) in enumerate(
        zip(switch_idxs, last_match_idxs)
    ):
        switch_frame_id = df.loc[switch_idx, "FrameId"]
        match_frame_id = df.loc[match_idx, "FrameId"]
        switch_hid = int(df.loc[switch_idx, "HId"])
        match_hid = int(df.loc[match_idx, "HId"])
        switch_img = sequence[switch_frame_id]["img"]
        match_img = sequence[match_frame_id]["img"]

        switch_box = track["boxes"][switch_frame_id + 1][switch_hid + 1]
        switch_box = torchvision.ops.box_convert(switch_box, "xyxy", "xywh")
        match_box = track["boxes"][match_frame_id + 1][match_hid + 1]
        match_box = torchvision.ops.box_convert(match_box, "xyxy", "xywh")

        ax[event, 0].imshow(match_img.permute(1, 2, 0))
        ax[event, 0].add_patch(
            Rectangle(
                match_box[:2], match_box[2], match_box[3], fill=False, color="r"
            )
        )
        ax[event, 0].tick_params(axis="both", labelsize=0, length=0)
        ax[event, 0].set_ylabel(f"frame_id: {match_frame_id}")

        ax[event, 1].imshow(switch_img.permute(1, 2, 0))
        ax[event, 1].add_patch(
            Rectangle(
                switch_box[:2],
                switch_box[2],
                switch_box[3],
                fill=False,
                color="g",
            )
        )
        ax[event, 1].tick_params(axis="both", labelsize=0, length=0)
        ax[event, 1].set_ylabel(f"frame_id: {switch_frame_id}")

    ax[0, 0].set_title("Last Detection before ID-Loss", fontsize=20)
    ax[0, 1].set_title("First Detection with new ID", fontsize=20)


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
        "switches": lambda show_switch_idx: visualize_sequence_switches(
            events_df, sequence, track, show_switch_idx
        ),
    }
    return visualize_functions
