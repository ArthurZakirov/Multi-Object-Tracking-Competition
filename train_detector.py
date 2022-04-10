import os
import json
import argparse
import time
from datetime import datetime
from collections import defaultdict
import random
import warnings
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from src.detector.object_detector import init_detector_from_config
from src.utils.file_utils import ensure_dir
from src.tracker.data_track import MOT16Sequences
from src.detector.data_obj_detect import MOT16ObjDetect
from src.detector.utils import (
    obj_detect_transforms,
    run_obj_detect,
    convert_frames,
)
from src.utils.train_utils import log_to_tensorboard

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_detect_config_path",
    type=str,
    default="config/obj_detect/fasterrcnn.json",
)
parser.add_argument("--tracker_or_detector_data", type=str, default="tracker")
parser.add_argument("--data_root_dir", type=str, default="data/MOT16")
parser.add_argument("--output_root_dir", type=str, default="models/obj_detect")
parser.add_argument("--train_split", type=str, default="train_wo_val2")
parser.add_argument("--eval_split", type=str, default="val2")
parser.add_argument("--train_sparse_version", action="store_true")
parser.add_argument("--eval_sparse_version", action="store_true")
parser.add_argument("--vis_threshold", type=float, default=0.25)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--eval_batch_frequency", type=int, default=1)
parser.add_argument("--save_batch_frequency", type=int, default=1)
parser.add_argument("--vis_batch_frequency", type=int, default=1)
args = parser.parse_args()


def main(args):

    if args.checkpoint_path is None:
        time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_dir = os.path.join(args.output_root_dir, time)
        summary_writer = SummaryWriter(output_dir)
        print(
            f"\n\nlook at tensorboard: \ntensorboard --logdir '{output_dir}'\n\n"
        )

        with open(args.obj_detect_config_path, "r") as f:
            hyperparams = json.load(f)

        output_model_config_path = os.path.join(output_dir, "model_config.json")
        ensure_dir(output_model_config_path)
        with open(output_model_config_path, "w") as f:
            json.dump(hyperparams, f)

        output_train_config_path = os.path.join(output_dir, "train_config.json")
        ensure_dir(output_train_config_path)
        with open(output_train_config_path, "w") as f:
            json.dump(args.__dict__, f)

        dataset_name = "MOT16"
        detector_name = hyperparams["name"]
        file_name = "best_" + detector_name + "_" + dataset_name + ".pth"

        obj_detect = init_detector_from_config(hyperparams)

        params = [p for p in obj_detect.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        lr_scheduler = None  # torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_AP = 0
        batch_count = 0

    else:
        output_dir = os.path.dirname(os.path.dirname(args.checkpoint_path))
        time = os.path.basename(output_dir)
        summary_writer = SummaryWriter(output_dir)

        print(
            f"\n\nlook at tensorboard: \ntensorboard --logdir '{output_dir}'\n\n"
        )

        model_config_path = os.path.join(output_dir, "model_config.json")
        with open(model_config_path, "r") as f:
            hyperparams = json.load(f)
        obj_detect = init_detector_from_config(hyperparams)

        train_config_path = os.path.join(output_dir, "train_config.json")
        with open(train_config_path, "r") as f:
            train_config = json.load(f)
        checkpoint_dict = torch.load(args.checkpoint_path)
        args = argparse.Namespace(**train_config)

        params = [p for p in obj_detect.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        lr_scheduler = None  # torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        obj_detect.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        best_AP = checkpoint_dict["best_AP"]
        batch_count = checkpoint_dict["batch_count"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj_detect.to(device)

    dataset_train = MOT16ObjDetect(
        root=args.data_root_dir,
        split=args.train_split,
        sparse_version=args.train_sparse_version,
        transforms=obj_detect_transforms(train=True),
        vis_threshold=args.vis_threshold,
        segmentation=hyperparams["return_segmentation"],
    )
    dataset_eval = MOT16ObjDetect(
        root=args.data_root_dir,
        split=args.eval_split,
        sparse_version=args.eval_sparse_version,
        transforms=obj_detect_transforms(train=False),
        vis_threshold=args.vis_threshold,
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    data_loader_eval = DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    for _ in trange(args.num_epochs, desc="epochs", leave=True):
        for images, targets in tqdm(
            data_loader_train, desc="train_batches", leave=False,
        ):
            obj_detect.train()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = obj_detect(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_to_tensorboard(
                metric_dict={
                    key: item.detach() for (key, item) in loss_dict.items()
                },
                step=batch_count,
                summary_writer=summary_writer,
                mode="train",
            )

            batch_count += 1

            if batch_count % args.eval_batch_frequency == 0:
                obj_detect.eval()
                detection_dict = run_obj_detect(
                    model=obj_detect,
                    data_loader=data_loader_eval,
                    debug=args.debug,
                )
                eval_dict = dataset_eval.evaluate_detections(detection_dict)
                log_to_tensorboard(
                    metric_dict=eval_dict,
                    step=batch_count,
                    summary_writer=summary_writer,
                    mode="eval",
                )
                if eval_dict["AP"] > best_AP:
                    best_AP = eval_dict["AP"]
                    save_model_path = os.path.join(
                        args.output_root_dir, time, file_name
                    )
                    ensure_dir(save_model_path)
                    torch.save(obj_detect, save_model_path)

            if batch_count % args.save_batch_frequency == 0:
                save_model_path = os.path.join(
                    args.output_root_dir,
                    time,
                    "checkpoint",
                    f"cpt_{batch_count}.pth",
                )
                ensure_dir(save_model_path)
                checkpoint_dir = {
                    "model": obj_detect.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "batch_count": batch_count,
                    "best_AP": best_AP,
                }
                torch.save(checkpoint_dir, save_model_path)

            if batch_count % args.vis_batch_frequency == 0:
                # TODO Visualization
                obj_detect.eval()
                with torch.no_grad():
                    pass


if __name__ == "__main__":
    main(args)
