#########################################
# Still ugly file with helper functions #
#########################################

import random
from collections import defaultdict
from os import path as osp
import csv

import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
from cycler import cycler as cy
from scipy.interpolate import interp1d
from torchvision.transforms import functional as F
import torchvision.transforms.functional as TF
import torchvision
from tqdm.auto import tqdm


colors = [
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "green",
    "greenyellow",
    "grey",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "rebeccapurple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]


def plot_sequence(tracks, db, first_n_frames=None):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
    """

    # print("[*] Plotting whole sequence to {}".format(output_dir))

    # if not osp.exists(output_dir):
    # 	os.makedirs(output_dir)

    # infinite color loop
    cyl = cy("ec", colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        img = v["img"].mul(255).permute(1, 2, 0).byte().numpy()
        width, height, _ = img.shape

        dpi = 96
        fig, ax = plt.subplots(1, dpi=dpi)
        fig.set_size_inches(width / dpi, height / dpi)
        ax.set_axis_off()
        ax.imshow(img)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0,
                        **styles[j]
                    )
                )

                ax.annotate(
                    j,
                    (
                        t_i[0] + (t_i[2] - t_i[0]) / 2.0,
                        t_i[1] + (t_i[3] - t_i[1]) / 2.0,
                    ),
                    color=styles[j]["ec"],
                    weight="bold",
                    fontsize=6,
                    ha="center",
                    va="center",
                )

        plt.axis("off")
        # plt.tight_layout()
        plt.show()
        # plt.savefig(im_output, dpi=100)
        # plt.close()

        if first_n_frames is not None and first_n_frames - 1 == i:
            break


def get_mot_accum(results, seq):
    """
    Arguments
    ---------
        results : results dict from Tracker object
        seq : MOT16Sequence object

    Returns
    -------
        mot_accum : MOTAccumulator object
            - contains tracking evaluation of a single sequence
    """
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i in range(len(seq)):
        gt = seq.data[i]["gt"]

        gt_ids = []
        gt_boxes = []
        for gt_id, box in gt.items():
            gt_ids.append(gt_id)
            gt_boxes.append(box)

        gt_boxes = np.stack(gt_boxes, axis=0)
        gt_boxes = torchvision.ops.box_convert(
            torch.from_numpy(gt_boxes), "xyxy", "xywh"
        ).numpy()

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            track_boxes = torchvision.ops.box_convert(
                torch.from_numpy(track_boxes), "xyxy", "xywh"
            ).numpy()
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(gt_ids, track_ids, distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,
    )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)
    return summary


def run_tracker(sequences, tracker):
    mot_accums = []
    results_seq = {}
    for sequence in tqdm(sequences, desc="sequences", leave=True):
        tracker.reset()

        sequence_loader = torch.utils.data.DataLoader(
            sequence, batch_size=1, shuffle=False
        )
        with torch.no_grad():
            for frame in tqdm(sequence, desc="frame", leave=False):
                tracker.step(frame)

        results_dict = tracker.get_results()
        results_seq[str(sequence)] = results_dict

        if not sequence.no_gt:
            mot_accums.append(get_mot_accum(results_dict, sequence))

    if mot_accums:
        print("\nevaluate_mot_accums...")
        eval_df = evaluate_mot_accums(
            accums=mot_accums,
            names=[
                str(sequence) for sequence in sequences if not sequence.no_gt
            ],
            generate_overall=True,
        )
    else:
        eval_df = None

    return eval_df, results_seq


def load_distance_fn(metric):
    if metric == "cosine_distance":
        return cosine_distance

    if metric == "euclidian_distance":
        pass
        # return euclidian_distance


def cosine_distance(input1, input2):
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = torch.nn.functional.normalize(input1, p=2, dim=1)
    input2_normed = torch.nn.functional.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def get_crop_from_boxes(boxes, image, output_height=256, output_width=128):
    """Crops all persons from a frame given the boxes.

    Args:
        boxes: The bounding boxes.
        image: The current image.
        height (int, optional): [description]. Defaults to 256.
        width (int, optional): [description]. Defaults to 128.
    """
    person_crops = []
    norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
    for box in boxes:
        box = box.to(torch.int32)
        res = image[:, :, box[1] : box[3], box[0] : box[2]]
        res = F.interpolate(res, (output_height, output_width), mode="bilinear")
        res = TF.normalize(res[0, ...], norm_mean, norm_std)
        person_crops.append(res.unsqueeze(0))
    return person_crops


def write_results(tracker_results_dict, output_path):
    """Write the tracks in the format for MOT16/MOT17 sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
    with open(output_path, "w", newline="") as of:
        writer = csv.writer(of, delimiter=",")
        for i, track in tracker_results_dict.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                score = int(np.round(bb[4].item()))
                writer.writerow(
                    [
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        score,
                        -1,
                        -1,
                        -1,
                    ]
                )

