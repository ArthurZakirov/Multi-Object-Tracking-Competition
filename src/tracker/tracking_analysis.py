import numpy as np
import pandas as pd


def groupSequence(lst):
    res = [[lst[0]]]

    for i in range(1, len(lst)):
        if lst[i - 1] + 1 == lst[i]:
            res[-1].append(lst[i])

        else:
            res.append([lst[i]])
    return res


def find_misses(events_df):
    """
    miss_seq_idxs
        list of lists of df indices, which together mark a sequence of misses of an object without matchs in between

    last_match_idxs
        list of lists of df indices, which together mark a the last sequence of matches, before the object was missed.
        More specifically, it's the last sequence of matches under the same track identity. 
        If the detected sequence contains an ID switch, then the saved sequence only has the elements of the last ID
        If it's an inherent miss, the list is empty.
    """
    miss_sequences = []
    last_match_sequences = []

    miss_ids = events_df.groupby("Type").get_group("MISS")["OId"].unique()
    for id in miss_ids:
        obj_df = events_df.groupby("OId").get_group(id).copy()
        obj_df.reset_index(drop=False, inplace=True)

        if obj_df.loc[0, "Type"] == "MISS":
            inherent_miss_mask = np.logical_and(
                obj_df["FrameId"] > 0, obj_df["Type"] == "MISS"
            )
            inherent_miss_df = obj_df[inherent_miss_mask].copy()
            inherent_miss_frame_ids = (
                inherent_miss_df["FrameId"].copy().tolist()
            )
            inherent_miss_frame_ids = groupSequence(inherent_miss_frame_ids)[0]
            inherent_miss_df.set_index("FrameId", inplace=True)
            inherent_miss_idxs = inherent_miss_df.loc[
                inherent_miss_frame_ids, "index"
            ].tolist()
            miss_sequences.append(inherent_miss_idxs)
            last_match_sequences.append([])

        miss_after_match_bool = 0 == np.convolve(
            (obj_df["Type"] != "MISS").astype(int) + 1, [-2, 1], "valid"
        )
        last_match_local_idxs = np.where(miss_after_match_bool)[0]

        for local_idx in last_match_local_idxs:
            global_idx = obj_df.loc[local_idx, "index"]
            frame_id = obj_df.loc[local_idx, "FrameId"]
            hid = obj_df.loc[local_idx, "HId"]

            lost_miss_mask = np.logical_and(
                obj_df["FrameId"] > frame_id, obj_df["Type"] == "MISS"
            )
            lost_miss_df = obj_df[lost_miss_mask].copy()
            lost_miss_frame_ids = lost_miss_df["FrameId"].copy().tolist()
            lost_miss_frame_ids = groupSequence(lost_miss_frame_ids)[0]
            lost_miss_df.set_index("FrameId", inplace=True)
            lost_miss_idxs = lost_miss_df.loc[
                lost_miss_frame_ids, "index"
            ].tolist()
            miss_sequences.append(lost_miss_idxs)

            last_match_mask = np.logical_and(
                np.logical_and(
                    obj_df["FrameId"] <= frame_id, obj_df["Type"] != "MISS"
                ),
                obj_df["HId"] == hid,
            )
            last_match_df = obj_df[last_match_mask].copy()
            last_match_frame_ids = last_match_df["FrameId"].copy().tolist()
            last_match_frame_ids = groupSequence(last_match_frame_ids)[0]
            last_match_df.set_index("FrameId", inplace=True)
            last_match_idxs = last_match_df.loc[
                last_match_frame_ids, "index"
            ].tolist()
            last_match_sequences.append(last_match_idxs)

    return miss_sequences, last_match_sequences


def find_switches(events_df):
    """
    switch_idxs: indices of the dataframe at which the switch happens

    last_match_idxs : indices of the dataframe at which the last match before the switch happens

    ascend_bool : boolean values for every switch event. 
        True: new track was started for obj. 
        False: obj was matched to a previously existing, but different track.
    
    active_switch_bool : boolean values for every switch event. 
        True: there are no misses between last match and switch 
        False: there are some misses between last match and switch 
    """
    switch_df = events_df.groupby("Type").get_group("SWITCH")
    switch_oids = switch_df["OId"].unique()
    switch_idxs = []
    last_match_idxs = []
    ascend_bool = []
    active_switch_bool = []
    for id in switch_oids:
        obj_df = events_df.groupby("OId").get_group(id)
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

            #
            # TODO : if not ascended, get indices of the wrong track id before the switch. will be used for visualization
            #

            last_match_idxs.append(last_match_before_switch_idx)
            switch_idxs.append(switch_idx)
            ascend_bool.append(
                not (
                    before_switch_df["HId"] == obj_df.loc[switch_idx, "HId"]
                ).any()
            )
            active_switch_bool.append(
                obj_df.loc[switch_idx, "FrameId"]
                - obj_df.loc[last_match_before_switch_idx, "FrameId"]
                == 1
            )
    return switch_idxs, last_match_idxs, ascend_bool, active_switch_bool


def find_fp(events_df):
    """
    fp_sequences: list of lists of df indices, which represent sequences of FP of a single Hypothesis ID
    """
    fp_sequences = []

    fp_df = events_df.groupby("Type").get_group("FP")
    fp_ids = fp_df["HId"].unique()
    for id in fp_ids:
        hyp_df = fp_df.groupby("HId").get_group(id).copy()
        hyp_frame_sequences = groupSequence(hyp_df["FrameId"].tolist())
        hyp_df.reset_index(drop=False, inplace=True)
        hyp_df.set_index("FrameId", inplace=True)
        for frame_seq in hyp_frame_sequences:
            idx_sequence = hyp_df.loc[frame_seq, "index"].tolist()
            fp_sequences.append(idx_sequence)
    return fp_sequences


def find_tp(events_df):
    """
    tp_sequences: list of lists of df indices, which represent sequences of MATCH or SWITCH of a single obj
    """
    tp_sequences = []

    tp_df = events_df[events_df["Type"].isin(["MATCH", "SWITCH"])].copy()
    tp_ids = tp_df["OId"].unique()
    for id in tp_ids:
        obj_df = tp_df.groupby("OId").get_group(id).copy()
        obj_frame_sequences = groupSequence(obj_df["FrameId"].tolist())
        obj_df.reset_index(drop=False, inplace=True)
        obj_df.set_index("FrameId", inplace=True)
        for frame_seq in obj_frame_sequences:
            idx_sequence = obj_df.loc[frame_seq, "index"].tolist()
            tp_sequences.append(idx_sequence)
    return tp_sequences


def get_sequences_items(
    sequence,
    events_df,
    idx_sequences,
    keys=["vis", "contrast", "luminosity", "area"],
):
    items = {key: [] for key in keys}
    for idxs in idx_sequences:
        first_idx = idxs[0]
        for key in keys:
            item = item_of_mot_event(sequence, events_df, first_idx, key=key)
            items[key].append(item)
    return items


def item_of_mot_event(sequence, events_df, df_idx, key="vis"):
    frame_id = events_df.loc[df_idx, "FrameId"]
    oid = events_df.loc[df_idx, "OId"]
    if key == "img":
        item = sequence[frame_id][key]
    else:
        item = sequence[frame_id][key][oid]
    return item


from collections import defaultdict


def split_items_by_vis(items, vis_tresholds=[0.0, 0.1, 0.3, 0.7, 1.0]):
    items_per_vis = defaultdict(dict)
    for i in range(len(vis_tresholds) - 1):
        mask = np.logical_and(
            np.array(items["vis"]) > vis_tresholds[i],
            np.array(items["vis"]) < vis_tresholds[i + 1],
        )
        for key in items.keys():
            items[key] = np.array(items[key])
            items_per_vis[vis_tresholds[i + 1]][key] = items[key][mask]
    return items_per_vis
