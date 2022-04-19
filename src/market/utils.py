from __future__ import division, absolute_import
import datetime
from tqdm import tqdm
from collections import defaultdict
import torch
import numpy as np

from . import metrics

__all__ = ["AverageMeter", "MetricMeter"]


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                "{} {:.4f} ({:.4f})".format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)


def extract_features(model, data_loader):
    f_, pids_, camids_ = [], [], []
    for data in tqdm(data_loader, desc="eval batches", leave=False):
        imgs, pids, camids = data["img"], data["pid"], data["camid"]
        imgs = imgs.to(list(model.parameters())[0].device)
        features = model(imgs)
        features = features.cpu().clone()
        f_.append(features)
        pids_.extend(pids)
        camids_.extend(camids)
    f_ = torch.cat(f_, 0)
    pids_ = np.asarray(pids_)
    camids_ = np.asarray(camids_)
    return f_, pids_, camids_


def print_statistics(
    batch_idx, num_batches, epoch, max_epoch, batch_time, losses
):
    batches_left = num_batches - (batch_idx + 1)
    future_batches_left = (max_epoch - (epoch + 1)) * num_batches
    eta_seconds = batch_time.avg * (batches_left + future_batches_left)
    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
    print(
        "epoch: [{0}/{1}][{2}/{3}]\t"
        "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        "eta {eta}\t"
        "{losses}\t".format(
            epoch + 1,
            max_epoch,
            batch_idx + 1,
            num_batches,
            batch_time=batch_time,
            eta=eta_str,
            losses=losses,
        )
    )
