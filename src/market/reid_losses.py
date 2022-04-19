import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tracker.utils import load_distance_fn


class GroupLossPrepare(object):
    def __init__(self, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

    def __call__(self, logits, features, gt_pids):
        # refine
        distance = torch.corrcoef(features)
        for _ in range(self.num_iterations):
            pi_support = torch.matmul(distance, logits)
            logits = F.normalize(logits * pi_support, p=1, dim=-1)

        # calculate loss
        return logits, gt_pids


class HardBatchMiningTripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining of samples in a batch.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, distance_metric="cosine_distance"):
        super(HardBatchMiningTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.distance_fn = load_distance_fn(metric=distance_metric)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """
        n = inputs.size(0)

        distance_matrix = self.distance_fn(inputs, inputs)
        distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()

        distance_positive_pairs = []
        distance_negative_pairs = []

        for i in range(n):
            positive = (targets[i] == targets).bool()
            negative = ~positive

            hard_positive = distance_matrix[i, positive].max()
            distance_positive_pairs.append(hard_positive)

            hard_negative = distance_matrix[i, negative].min()
            distance_negative_pairs.append(hard_negative)

        distance_positive_pairs = torch.Tensor(distance_positive_pairs)
        distance_negative_pairs = torch.Tensor(distance_negative_pairs)

        y = torch.ones_like(distance_negative_pairs)
        return self.ranking_loss(
            distance_negative_pairs, distance_positive_pairs, y
        )


class CombinedLoss(object):
    def __init__(
        self,
        margin=0.3,
        distance_metric="cosine_distance",
        num_iterations=3,
        weight_crossentropy=0.5,
        weight_triplet=0.5,
    ):
        super().__init__()
        self.group_loss_prepare = GroupLossPrepare(num_iterations)
        self.triplet_loss = HardBatchMiningTripletLoss(margin, distance_metric)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.weight_triplet = weight_triplet
        self.weight_crossentropy = weight_crossentropy

    @classmethod
    def from_config(cls, hyperparams):
        self = cls.__new__(cls)
        self.__init__(**hyperparams)
        return self

    def __call__(self, logits, features, gt_pids):
        loss = 0.0
        loss_summary = {}
        if self.weight_triplet > 0.0:
            loss_t = self.triplet_loss(features, gt_pids) * self.weight_triplet
            loss += loss_t
            loss_summary["Triplet Loss"] = loss_t

        if self.weight_crossentropy > 0.0:
            logits, gt_pids = self.group_loss_prepare(logits, features, gt_pids)
            loss_ce = (
                self.cross_entropy(logits, gt_pids) * self.weight_crossentropy
            )
            loss += loss_ce
            loss_summary["CE Loss"] = loss_ce

        loss_summary["Loss"] = loss
        return loss, loss_summary
