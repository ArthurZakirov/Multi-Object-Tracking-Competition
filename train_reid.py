from math import hypot
import os
from datetime import datetime
import json
import argparse
import time
from tqdm import trange, tqdm
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from src.market.datamanager import ImageDataManager
from src.market.models import build_model
from src.market import utils
from src.market import metrics
from src.market.reid_losses import CombinedLoss
from src.tracker.utils import load_distance_fn
from src.utils.file_utils import ensure_dir
from src.utils.train_utils import log_to_tensorboard

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reid_model_config_path",
    type=str,
    default="config/reid_model/reid_on_seg.json",
    help="path to model dir",
)
parser.add_argument(
    "--output_root_dir",
    type=str,
    default="models/reid_model",
    help="path to model dir",
)
parser.add_argument("--data_root_dir", type=str, default="data/market")

parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size_train", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--amsgrad", type=int, default=1)
parser.add_argument("--lr_schedulder_step_size", type=int, default=10)


parser.add_argument("--eval_frequency", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


def evaluate_reid_model(model, distance_fn, test_loader, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.eval()
        print("\nExtracting features from query set...")
        q_feat, q_pids, q_camids = utils.extract_features(
            model, test_loader["query"]
        )
        print(
            "\nDone, obtained {}-by-{} matrix".format(
                q_feat.size(0), q_feat.size(1)
            )
        )

        print("\nExtracting features from gallery set ...")
        g_feat, g_pids, g_camids = utils.extract_features(
            model, test_loader["gallery"]
        )
        print(
            "\nDone, obtained {}-by-{} matrix".format(
                g_feat.size(0), g_feat.size(1)
            )
        )

        distmat = distance_fn(q_feat, g_feat).numpy()

        print("Computing CMC and mAP ...")
        cmc, mAP = metrics.eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50
        )

        print("** Results **")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        return cmc[0], mAP


def main():
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_dir = os.path.join(args.output_root_dir, time)
    summary_writer = SummaryWriter(output_dir)

    print(f"\n\nlook at tensorboard: \ntensorboard --logdir '{output_dir}'\n\n")
    with open(args.reid_model_config_path, "r") as f:
        hyperparams = json.load(f)

    output_model_config_path = os.path.join(output_dir, "model_config.json")
    ensure_dir(output_model_config_path)
    with open(output_model_config_path, "w") as f:
        json.dump(hyperparams, f)

    output_training_config_path = os.path.join(output_dir, "train_config.json")
    ensure_dir(output_training_config_path)
    with open(output_training_config_path, "w") as f:
        json.dump(args.__dict__, f)

    datamanager = ImageDataManager(
        root=args.data_root_dir,
        masked=hyperparams["input_is_masked"],
        height=hyperparams["height"],
        width=hyperparams["width"],
        batch_size_train=args.batch_size_train,
        workers=args.num_workers,
        transforms=["random_flip", "random_crop"],
    )

    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    model = build_model(
        hyperparams["backbone"],
        num_classes=datamanager.num_train_pids,
        loss=hyperparams["loss"],
        pretrained=hyperparams["pretrained"],
        input_is_masked=hyperparams["input_is_masked"],
    )
    distance_fn = load_distance_fn(
        hyperparams["loss_hyperparams"]["distance_metric"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_schedulder_step_size
    )
    criterion = CombinedLoss.from_config(hyperparams["loss_hyperparams"])

    num_batches = len(train_loader)
    best_mAP = 0

    for epoch in trange(args.num_epochs, desc="epoch", leave=True):
        model.train()
        for batch_idx, data in tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc="train batch",
            leave=False,
        ):
            # Predict output.
            imgs = data["img"].to(device)
            pids = data["pid"].to(device)
            logits, features = model(imgs)
            # Compute loss.
            loss, loss_summary = criterion(logits, features, pids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_to_tensorboard(
                metric_dict=loss_summary,
                step=epoch * num_batches + batch_idx,
                summary_writer=summary_writer,
                mode="train",
            )

            if args.debug:
                break

        scheduler.step()
        if (
            epoch + 1
        ) % args.eval_frequency == 0 or epoch == args.num_epochs - 1:
            rank1, mAP = evaluate_reid_model(model, distance_fn, test_loader)
            log_to_tensorboard(
                metric_dict={"rank1": rank1, "mAP": mAP},
                step=epoch,
                summary_writer=summary_writer,
                mode="eval",
            )
            if mAP > best_mAP:
                best_mAP = mAP
                checkpoint_path = os.path.join(
                    output_dir, "checkpoints", f"cpt_{str(epoch)}.pth"
                )
                ensure_dir(checkpoint_path)
                torch.save(model, checkpoint_path)


if __name__ == "__main__":
    main()

