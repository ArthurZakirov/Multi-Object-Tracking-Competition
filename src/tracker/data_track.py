import configparser
import csv
from collections import defaultdict
from msilib import sequence
import os
import os.path as osp
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from src.utils.file_utils import ensure_dir
from src.detector.utils import decode_segmentation

_sets = {}
splits = [
    "mini",
    "train",
    "test",
    "all",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "reid",
    "train_wo_val",
    "train_wo_val2",
    "val",
    "val2",
]

# Fill all available datasets, change here to modify / add new datasets.
for split in splits:
    dataset_name = f"MOT16-{split}"
    _sets[dataset_name] = lambda root_dir, split=split, **kwargs: MOT16(
        root_dir, split, **kwargs
    )


def split_sequence_names(split, root_dir):
    train_sequences = list(listdir_nohidden(os.path.join(root_dir, "train")))
    test_sequences = ["MOT16-01", "MOT16-03", "MOT16-08", "MOT16-12"]
    val_sequences = [
        "MOT16-02",
        "MOT16-05",
        "MOT16-09",
        "MOT16-11",
    ]  # these all contain segmentation
    val_sequences2 = ["MOT16-02", "MOT16-11"]

    if "train" == split:
        sequences = train_sequences
    elif "test" == split:
        sequences = test_sequences
    elif "all" == split:
        sequences = train_sequences + test_sequences
    elif "reid" == split:
        sequences = ["MOT16-02", "MOT16-05", "MOT16-09", "MOT16-11"]
    elif "train_wo_val" == split:
        sequences = [seq for seq in train_sequences if seq not in val_sequences]
    elif "train_wo_val2" == split:
        sequences = [
            seq for seq in train_sequences if seq not in val_sequences2
        ]
    elif "val" == split:
        sequences = val_sequences
    elif "val2" == split:
        sequences = val_sequences2
    elif f"MOT16-{split}" in train_sequences + test_sequences:
        sequences = [f"MOT16-{split}"]
    elif "mini" == split:
        sequences = ["MOT16-02-mini"]
    else:
        raise NotImplementedError("MOT split not available.")
    return sequences


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f


class MOT16Sequences:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, dataset, root_dir, **kwargs):
        """Initialize the corresponding dataloader.

        Keyword arguments:
        dataset --  the name of the dataset
        args -- arguments used to call the dataset
        """
        assert dataset in _sets, "[!] Dataset not found: {}".format(dataset)

        self._data = _sets[dataset](root_dir, **kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MOT16(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, root_dir, split, **kwargs):
        """Initliazes all subset of the dataset.

        Keyword arguments:
        root_dir -- directory of the dataset
        split -- the split of the dataset to use
        args -- arguments used to call the dataset
        """
        sequence_names = split_sequence_names(split, root_dir)

        self._data = []
        for sequence_name in sequence_names:
            sequence = MOT16Sequence(root_dir, seq_name=sequence_name, **kwargs)
            self._data.append(sequence)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MOT16Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataset is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(
        self,
        root_dir,
        seq_name,
        vis_threshold=0.0,
        reid_on_gt_model_name=None,
        return_gt_segmentation=False,
        only_obj_w_mask=True,
    ):
        """
        Args:
            root_dir -- directory of the dataset
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        self._return_gt_segmentation = return_gt_segmentation
        self._only_obj_w_mask = only_obj_w_mask
        self._mot_dir = root_dir

        self._train_folders = os.listdir(os.path.join(self._mot_dir, "train"))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, "test"))

        self.transforms = ToTensor()

        assert (
            seq_name in self._train_folders or seq_name in self._test_folders
        ), "Image set does not exist: {}".format(seq_name)

        self.data, self.no_gt = self._sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]

        img = Image.open(data["im_path"]).convert("RGB")

        img = self.transforms(img)

        sample = {}
        sample["img"] = img
        sample["img_path"] = data["im_path"]
        sample["gt"] = data["gt"]
        sample["vis"] = data["vis"]

        # segmentation
        if data["seg_img"] is not None:
            seg_img = np.array(data["seg_img"])
            seg_img = decode_segmentation(seg_img)
            sample["seg_img"] = seg_img

        return sample

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, "train", seq_name)
        else:
            seq_path = osp.join(self._mot_dir, "test", seq_name)

        config_file = osp.join(seq_path, "seqinfo.ini")

        assert osp.exists(config_file), "Config file does not exist: {}".format(
            config_file
        )

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config["Sequence"]["seqLength"])
        img_dir = config["Sequence"]["imDir"]

        img_dir = osp.join(seq_path, img_dir)
        gt_file = osp.join(seq_path, "gt", "gt.txt")
        seg_dir = osp.join(seq_path, "seg_ins")

        data = []
        boxes = {}
        visibility = {}
        seg_imgs = {}

        # for i in range(1, seqLength + 1):
        #     boxes[i] = {}
        #     visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            gt = load_detection_from_txt(
                txt_path=gt_file, vis_threshold=self._vis_threshold, mode="gt"
            )
            visibility = gt["visibilities"]
            boxes = gt["boxes"]
        else:
            no_gt = True

        if self._return_gt_segmentation:
            if osp.exists(seg_dir):
                for seg_file in listdir_nohidden(seg_dir):
                    frame_id = int(seg_file.split(".")[0])
                    seg_img = Image.open(osp.join(seg_dir, seg_file))
                    seg_imgs[frame_id] = seg_img

        for img_file in os.listdir(img_dir):
            img_path = osp.join(img_dir, img_file)
            i = int(img_file.split(".")[0])

            datum = {
                "gt": boxes[i],
                "im_path": img_path,
                "vis": visibility[i],
            }

            datum["seg_img"] = None
            if seg_imgs:
                datum["seg_img"] = seg_imgs[i]

            data.append(datum)

        return data, no_gt

    def __str__(self):
        return self._seq_name

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        file = osp.join(output_dir, "MOT16-" + self._seq_name[6:8] + ".txt")
        ensure_dir(file)
        print("Writing predictions to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=",")
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [
                            frame + 1,
                            i + 1,
                            x1 + 1,
                            y1 + 1,
                            x2 - x1 + 1,
                            y2 - y1 + 1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ]
                    )


def load_detection_from_txt(txt_path, vis_threshold=0.0, mode="gt"):
    """
    modes: "det", "gt", "track"
    """
    boxes = defaultdict(dict)
    visibility = defaultdict(dict)
    gt = {}
    with open(txt_path, "r") as inf:
        reader = csv.reader(inf, delimiter=",")
        for row in reader:
            # class person, certainity 1, visibility >= 0.25
            if mode == "gt" and not (
                int(row[6]) == 1
                and int(row[7]) == 1
                and float(row[8]) >= vis_threshold
            ):
                continue
            elif mode == "gt":
                visibility[int(row[0])][int(row[1])] = float(row[8])

            if mode == "det" and False:  # TODO add condition
                continue
            if mode == "track" and False:  # TODO add condition
                continue
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = float(row[2]) - 1
            y1 = float(row[3]) - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + float(row[4]) - 1
            y2 = y1 + float(row[5]) - 1
            bb = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            boxes[int(row[0])][int(row[1])] = bb

    gt["boxes"] = boxes
    if mode == "gt":
        gt["visibilities"] = visibility
    return gt

