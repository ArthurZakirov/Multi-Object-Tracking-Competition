import os
import json
import argparse
from datetime import datetime
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from src.tracker.data_track_precomputed import MOT16SequencesPrecomputed

from src.tracker.utils import run_tracker
from src.gnn.assignment_model import AssignmentSimilarityNet
from src.gnn.dataset import LongTrackTrainingDataset
from src.tracker.tracker import MyTracker
from src.tracker.data_track import MOT16Sequences
from src.utils.file_utils import ensure_dir

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--assign_model_config_path",
    type=str,
    default="config/assign_model/gnn.json",
    help="path to model config",
)
parser.add_argument(
    "--output_root_dir",
    type=str,
    default="models/assign_model",
    help="path to model dir",
)
parser.add_argument(
    "--data_root_dir",
    type=str,
    default="data/MOT16",
    help="path to evaluation data root",
)
parser.add_argument(
    "--eval_split",
    type=str,
    default="val2",
    help="part of dataset, choose from ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', 'reid', 'train_wo_val', 'train_wo_val2', 'val', 'val2']",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train_wo_val2",
    help="part of dataset, choose from ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', 'reid', 'train_wo_val', 'train_wo_val2', 'val', 'val2']",
)
parser.add_argument(
    "--precomputed_data_root_dir",
    type=str,
    default="data/precomputed_detection/default",
)
parser.add_argument(
    "--reid_on_det_model_name", type=str, default="reid",
)
parser.add_argument(
    "--reid_on_gt_model_name", type=str, default="reid",
)

parser.add_argument(
    "--vis_threshold",
    type=float,
    default=0.0,
    help="Threshold of visibility of persons above which they are selected",
)
parser.add_argument(
    "--eval_tracker_config_path",
    type=str,
    default="config/tracker/tracker.json",
    help="path to tracker configuration",
)


parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--lr_scheduler_step_size", type=float, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug", action="store_true")


args = parser.parse_args()


@torch.no_grad()
def compute_class_metric(
    pred, target, class_metrics=("accuracy", "recall", "precision")
):
    TP = ((target == 1) & (pred == 1)).sum().float()
    FP = ((target == 0) & (pred == 1)).sum().float()
    TN = ((target == 0) & (pred == 0)).sum().float()
    FN = ((target == 1) & (pred == 0)).sum().float()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
    precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict = {
        "accuracy": accuracy.item(),
        "recall": recall.item(),
        "precision": precision.item(),
    }
    class_metrics_dict = {
        met_name: class_metrics_dict[met_name] for met_name in class_metrics
    }

    return class_metrics_dict


def log_to_tensorboard(metric_dict, step, summary_writer, mode="train"):
    for metric_name, metric_value in metric_dict.items():
        summary_writer.add_scalar(
            tag=os.path.join(mode, metric_name),
            scalar_value=metric_value,
            global_step=step,
        )


def train_one_epoch(args, model, data_loader, optimizer, epoch, summary_writer):
    model.train()
    device = next(model.parameters()).device
    metrics_accum = {
        "loss": 0.0,
        "accuracy": 0.0,
        "recall": 0.0,
        "precision": 0.0,
    }

    for batch_idx, batch in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc="batches",
        leave=False,
    ):
        optimizer.zero_grad()

        # Since our model does not support automatic batching, we do manual
        # gradient accumulation
        for sample in batch:
            past_frame, curr_frame = sample
            track_feats, track_coords, track_ids = (
                past_frame["features"].to(device),
                past_frame["boxes"].to(device),
                past_frame["ids"].to(device),
            )
            current_feats, current_coords, curr_ids = (
                curr_frame["features"].to(device),
                curr_frame["boxes"].to(device),
                curr_frame["ids"].to(device),
            )
            track_t, curr_t = (
                past_frame["time"].to(device),
                curr_frame["time"].to(device),
            )
            track_masks = (
                None
                if not "masks" in past_frame.keys()
                else past_frame["masks"].to(device)
            )
            current_masks = (
                None
                if not "masks" in curr_frame.keys()
                else curr_frame["masks"].to(device)
            )
            assign_sim = model.forward(
                track_features=track_feats,
                current_features=current_feats,
                track_boxes=track_coords,
                current_boxes=current_coords,
                track_time=track_t,
                current_time=curr_t,
                track_masks=track_masks,
                current_masks=current_masks,
            )

            same_id = (track_ids.view(-1, 1) == curr_ids.view(1, -1)).type(
                assign_sim.dtype
            )
            same_id = same_id.unsqueeze(0).expand(assign_sim.shape[0], -1, -1)

            loss = F.binary_cross_entropy_with_logits(
                assign_sim, same_id, pos_weight=torch.as_tensor(20.0)
            ) / float(len(batch))
            loss.backward()

            # Keep track of metrics
            with torch.no_grad():
                pred = (assign_sim[-1] > 0.5).view(-1).float()
                target = same_id[-1].view(-1)
                metrics = compute_class_metric(pred, target)

                for m_name, m_val in metrics.items():
                    metrics_accum[m_name] += m_val / float(len(batch))
                metrics_accum["loss"] += loss.item()

            log_to_tensorboard(
                metric_dict=metrics_accum,
                step=epoch * len(data_loader) + batch_idx,
                summary_writer=summary_writer,
                mode="train",
            )

            metrics_accum = {
                "loss": 0.0,
                "accuracy": 0.0,
                "recall": 0.0,
                "precision": 0.0,
            }

        optimizer.step()
        if args.debug:
            break
    model.eval()


def evaluate_one_epoch(args, assign_model):
    with open(args.eval_tracker_config_path, "r") as f:
        tracker_hyperparams = json.load(f)
    tracker = MyTracker.from_config(tracker_hyperparams)
    tracker.assign_model = assign_model

    sequences = MOT16SequencesPrecomputed(
        precomputed_data_root_dir=args.precomputed_data_root_dir,
        original_data_root_dir=args.data_root_dir,
        split=args.eval_split,
        vis_threshold=args.vis_threshold,
        reid_on_det_model_name=args.reid_on_det_model_name,
        return_det_segmentation=assign_model.use_segmentation,
    )

    eval_df, _ = run_tracker(sequences=sequences, tracker=tracker)
    idf1 = eval_df.loc["OVERALL"]["idf1"]
    return idf1


def main():
    # files
    #########################################################################
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_dir = os.path.join(args.output_root_dir, time)

    summary_writer = SummaryWriter(output_dir)
    print(
        f"\n\n\look at tensorboard: \ntensorboard --logdir '{output_dir}'\n\n"
    )

    output_model_config_path = os.path.join(output_dir, "model_config.json")
    with open(args.assign_model_config_path, "r") as f:
        assign_model_hyperparams = json.load(f)

    ensure_dir(output_model_config_path)
    with open(output_model_config_path, "w") as f:
        json.dump(assign_model_hyperparams, f)

    training_hyperparams = {
        "learning_rate": args.learning_rate,
        "lr_scheduler_step_size": args.lr_scheduler_step_size,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
    }
    output_training_config_path = os.path.join(
        output_dir, "training_config.json"
    )
    with open(output_training_config_path, "w") as f:
        json.dump(training_hyperparams, f)

    # load everything
    #########################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assign_model = AssignmentSimilarityNet.from_config(assign_model_hyperparams)
    assign_model.to(device)

    train_dataset = LongTrackTrainingDataset(
        dataset=f"MOT16-{args.train_split}",
        root_dir=args.data_root_dir,
        precomputed_root_dir=args.precomputed_data_root_dir,
        reid_on_gt_model_name=args.reid_on_gt_model_name,
        max_past_frames=args.patience,
        vis_threshold=args.vis_threshold,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda x: x,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(
        assign_model.parameters(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_scheduler_step_size
    )

    # run training
    #########################################################################

    best_idf1 = 0.0
    for epoch in trange(1, args.num_epochs + 1, desc="epochs", leave=True):
        train_one_epoch(
            args=args,
            model=assign_model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            epoch=epoch,
            summary_writer=summary_writer,
        )
        scheduler.step()

        if (epoch % args.eval_every) == 0:
            idf1 = evaluate_one_epoch(args, assign_model)
            log_to_tensorboard(
                metric_dict={"idf1": idf1},
                step=epoch,
                summary_writer=summary_writer,
                mode="eval",
            )
            if idf1 > best_idf1:
                best_idf1 = idf1
                save_model_path = os.path.join(output_dir, "best_ckpt.pth")
                torch.save(assign_model, save_model_path)


if __name__ == "__main__":
    main()

