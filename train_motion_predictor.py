import os
import json
from datetime import datetime
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from src.motion_prediction.dataset import MOT16MotionPrediction
from src.motion_prediction.model import MotionPredictor
from src.utils.file_utils import ensure_dir
from src.utils.train_utils import log_to_tensorboard
from train_gnn import train_one_epoch


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--motion_predictor_config_path",
    type=str,
    default="config/motion_predictor/motion_predictor.json",
    help="path to model config",
)
parser.add_argument(
    "--output_root_dir",
    type=str,
    default="models/motion_predictor",
    help="path to model dir",
)
parser.add_argument(
    "--data_root_dir",
    type=str,
    default="data/MOT16",
    help="path to evaluation data root",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train_wo_val2",
    help="part of dataset, choose from ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', 'reid', 'train_wo_val', 'train_wo_val2', 'val', 'val2']",
)
parser.add_argument(
    "--eval_split",
    type=str,
    default="val2",
    help="part of dataset, choose from ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', 'reid', 'train_wo_val', 'train_wo_val2', 'val', 'val2']",
)


parser.add_argument(
    "--vis_threshold",
    type=float,
    default=0.0,
    help="Threshold of visibility of persons above which they are selected",
)


parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--lr_scheduler_step_size", type=float, default=5)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--vis_every", type=int, default=1)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug", action="store_true")


args = parser.parse_args()


def train_one_epoch(
    model, optimizer, loss_fn, dataloader, epoch, summary_writer, debug
):
    model.train()
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="train batches",
        leave=False,
    ):
        hist, fut = batch
        pred = model(hist)
        loss = loss_fn(pred, fut)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_to_tensorboard(
            metric_dict={"loss": loss.detach().item()},
            step=epoch * len(dataloader) + batch_idx,
            summary_writer=summary_writer,
            mode="train",
        )

        if debug:
            break


def evaluate_one_epoch(model, dataloader, loss_fn, debug):
    model.eval()
    eval_loss = []
    for batch in tqdm(dataloader, desc="eval batches", leave=False):
        hist, fut = batch
        with torch.no_grad():
            pred = model(hist)
        loss = loss_fn(pred, fut)
        eval_loss.append(loss)

        if debug:
            break
    return {"loss": torch.tensor(eval_loss).mean().item()}


def visualize_one_epoch(model, dataset, num_samples=1):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )
    iterator = iter(dataloader)

    fig, ax = plt.subplots(1, num_samples, squeeze=False)
    for i in range(num_samples):
        hist, fut = next(iterator)
        with torch.no_grad():
            pred = model(hist)

        hist = hist.squeeze(0)
        fut = fut.squeeze(0)
        pred = pred.squeeze(0)
        hist_xywh = torchvision.ops.box_convert(hist, "cxcywh", "xywh")
        fut_xywh = torchvision.ops.box_convert(fut, "cxcywh", "xywh")
        pred_xywh = torchvision.ops.box_convert(pred, "cxcywh", "xywh")
        ax[0, i].add_patch(
            Rectangle(
                hist_xywh[0, :2],
                hist_xywh[0, 2],
                hist_xywh[0, 3],
                fill=False,
                color="k",
            )
        )
        ax[0, i].add_patch(
            Rectangle(
                fut_xywh[-1, :2],
                fut_xywh[-1, 2],
                fut_xywh[-1, 3],
                fill=False,
                color="b",
            )
        )
        ax[0, i].add_patch(
            Rectangle(
                pred_xywh[-1, :2],
                pred_xywh[-1, 2],
                pred_xywh[-1, 3],
                fill=False,
                color="r",
            )
        )

        ax[0, i].plot(hist[:, 0], hist[:, 1], label="history", color="k")
        ax[0, i].plot(fut[:, 0], fut[:, 1], label="future", color="b")
        ax[0, i].plot(pred[:, 0], pred[:, 1], label="prediction", color="r")

        ax[0, i].grid()
        ax[0, i].axis("equal")

    ax[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return {"prediction": fig}


def main():
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_dir = os.path.join(args.output_root_dir, time)

    summary_writer = SummaryWriter(output_dir)

    print(f"\n\nlook at tensorboard: \ntensorboard --logdir '{output_dir}'\n\n")

    with open(args.motion_predictor_config_path, "r") as f:
        hyperparams = json.load(f)

    output_model_config_dir = os.path.join(output_dir, "model_config.json")
    ensure_dir(output_model_config_dir)
    with open(output_model_config_dir, "w") as f:
        json.dump(hyperparams, f)

    output_train_config_dir = os.path.join(output_dir, "train_config.json")
    ensure_dir(output_train_config_dir)
    with open(output_train_config_dir, "w") as f:
        json.dump(args.__dict__, f)

    model = MotionPredictor.from_config(hyperparams)

    train_dataset = MOT16MotionPrediction(
        root=args.data_root_dir,
        split=args.train_split,
        history_len=hyperparams["history_len"],
        future_len=hyperparams["future_len"],
    )
    train_dataset = torch.utils.data.Subset(
        train_dataset, torch.randperm(len(train_dataset)).tolist()[:10]
    )

    eval_dataset = MOT16MotionPrediction(
        root=args.data_root_dir,
        split=args.eval_split,
        history_len=hyperparams["history_len"],
        future_len=hyperparams["future_len"],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate
    )

    for epoch in trange(1, args.num_epochs + 1, desc="epochs", leave=True):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            epoch=epoch,
            summary_writer=summary_writer,
            debug=args.debug,
        )
        log_dict = {}
        if epoch % args.eval_every == 0:
            log_metrics_dict = evaluate_one_epoch(
                model=model,
                dataloader=eval_dataloader,
                loss_fn=loss_fn,
                debug=args.debug,
            )
            log_dict.update(log_metrics_dict)

        if epoch % args.vis_every == 0:
            log_vis_dict = visualize_one_epoch(
                model=model, dataset=train_dataset, num_samples=1
            )
            log_dict.update(log_vis_dict)
        log_to_tensorboard(
            metric_dict=log_dict,
            step=epoch,
            summary_writer=summary_writer,
            mode="eval",
        )


if __name__ == "__main__":
    main()
