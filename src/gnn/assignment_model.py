import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

from src.tracker.utils import load_distance_fn


def binary_mask_iou(masks1, masks2, box_iou=None):
    """
  Arguments
  ---------
    masks1 : [N1, H, W]
    masks2 : [N2, H, W]
    box_iou : [N1, N2] can be passed for speedup

  Returns
  -------
    iou_matrix : [N1, N2]
  """
    masks1_area = np.count_nonzero(masks1, axis=(-2, -1))
    masks2_area = np.count_nonzero(masks2, axis=(-2, -1))
    (masks1_area_matrix, masks2_area_matrix) = np.meshgrid(
        masks2_area, masks1_area
    )

    intersection_matrix = []
    for row, mask1 in enumerate(masks1):
        if not box_iou is None:
            intersection_row = np.zeros((len(masks2)))
            notnull = np.where(box_iou[row] > 0)[0]
            mask1 = np.tile(mask1, (len(notnull), 1, 1))
            notnull_intersection_row = np.count_nonzero(
                np.logical_and(mask1, masks2[notnull]), axis=(-2, -1)
            )
            intersection_row[notnull] = notnull_intersection_row
        else:
            mask1 = np.tile(mask1, (len(masks2), 1, 1))
            intersection_row = np.count_nonzero(
                np.logical_and(mask1, masks2), axis=(-2, -1)
            )

        intersection_matrix.append(intersection_row)
    intersection_matrix = np.stack(intersection_matrix, axis=0)
    union_matrix = masks1_area_matrix + masks2_area_matrix - intersection_matrix

    iou_matrix = intersection_matrix / union_matrix
    return iou_matrix


class BipartiteNeuralMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.0):
        super().__init__()

        edge_in_dim = 2 * node_dim + 2 * edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        node_in_dim = node_dim + edge_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.aggregate = lambda edge_embeds, dim: edge_embeds.sum(dim=dim)

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Node-to-edge updates, as descibed in slide 71, lecture 5.
        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, 2 x edge_dim) 
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
            
        returns:
            updated_edge_feats = torch.Tensor with shape (|A|, |B|, edge_dim) 
        """

        n_nodes_a, n_nodes_b, _ = edge_embeds.shape

        nodes_a_embeds = nodes_a_embeds.repeat(n_nodes_b, 1, 1).permute(1, 0, 2)
        nodes_b_embeds = nodes_b_embeds.repeat(n_nodes_a, 1, 1)

        edge_in = torch.cat(
            [nodes_a_embeds, nodes_b_embeds, edge_embeds], axis=-1
        )
        # has shape (|A|, |B|, 2*node_dim + 2*edge_dim)

        return self.edge_mlp(edge_in)

    def node_update(
        self, updatable_edge_embeds, nodes_a_embeds, nodes_b_embeds
    ):
        """
        Edge-to-node updates, as descibed in slide 75, lecture 5.

        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, edge_dim) 
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
            
        returns:
            tuple(
                updated_nodes_a_embeds: torch.Tensor with shape (|A|, node_dim),
                updated_nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
                )
        """

        message_to_a = self.aggregate(updatable_edge_embeds, dim=1)
        message_to_b = self.aggregate(updatable_edge_embeds, dim=0)

        nodes_a_in = torch.cat(
            [nodes_a_embeds, message_to_a], dim=-1
        )  # Has shape (|A|, node_dim + edge_dim)
        nodes_b_in = torch.cat(
            [nodes_b_embeds, message_to_b], dim=-1
        )  # Has shape (|B|, node_dim + edge_dim)

        nodes_a_embeds = self.node_mlp(nodes_a_in)
        nodes_b_embeds = self.node_mlp(nodes_b_in)

        return nodes_a_embeds, nodes_b_embeds

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        updatable_edge_embeds = self.edge_update(
            edge_embeds, nodes_a_embeds, nodes_b_embeds
        )
        nodes_a_embeds, nodes_b_embeds = self.node_update(
            updatable_edge_embeds, nodes_a_embeds, nodes_b_embeds
        )

        return updatable_edge_embeds, nodes_a_embeds, nodes_b_embeds


class AssignmentModel(nn.Module):
    def __init__(self, use_segmentation=False):
        super().__init__()
        self.use_segmentation = use_segmentation

    @classmethod
    def from_config(cls, hyperparams):
        self = cls.__new__(cls)
        self.__init__(**hyperparams)
        return self

    def forward(
        self,
        track_features,
        current_features,
        track_boxes,
        current_boxes,
        track_time,
        current_time,
        track_masks,
        current_masks,
    ):
        raise NotImplementedError()


class AssignmentWeightedAverage(AssignmentModel):
    def __init__(
        self, combine_weights, distance_metric, use_segmentation=False
    ):
        super().__init__(use_segmentation)
        self.combine_weights = combine_weights
        self.distance_fn = load_distance_fn(distance_metric)

    def forward(
        self,
        track_features,
        current_features,
        track_boxes,
        current_boxes,
        track_time,
        current_time,
        track_masks,
        current_masks,
    ):
        combined_distance_matrix = []
        for metric, weight in self.combine_weights.items():
            if weight == 0:
                continue

            if metric == "box_iou_distance" and not current_boxes.isnan().any():
                box_iou = torchvision.ops.box_iou(track_boxes, current_boxes)
                distance_matrix = 1 - box_iou

            if (
                metric == "mask_iou_distance"
                and not current_masks.isnan().any()
            ):
                mask_iou = binary_mask_iou(
                    track_masks, current_masks, box_iou=box_iou
                )
                distance_matrix = 1 - mask_iou

            if metric == "reid_distance" and not current_features.isnan().any():
                distance_matrix = self.metric_fn(
                    track_features, current_features
                )

            combined_distance_matrix.append(weight * distance_matrix)

        combined_distance_matrix = np.stack(
            combined_distance_matrix, axis=0
        ).sum(axis=0)

        return combined_distance_matrix


class AssignmentSimilarityNet(AssignmentModel):
    def __init__(
        self,
        node_dim,
        edge_dim,
        reid_dim,
        edges_in_dim,
        num_steps,
        distance_metric,
        dropout=0.0,
        use_segmentation=False,
    ):
        super().__init__(use_segmentation)
        self.distance_fn = load_distance_fn(distance_metric)
        self.graph_net = BipartiteNeuralMessagePassingLayer(
            node_dim=node_dim, edge_dim=edge_dim, dropout=dropout
        )
        self.num_steps = num_steps
        self.linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(
            nn.Linear(edges_in_dim, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)
        )

    def compute_edge_feats(self, track_coords, current_coords, track_t, curr_t):
        """
        Computes initial edge feature tensor

        Args:
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)
                          
            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)        
            
        
        Returns:
            tensor with shape (num_trakcs, num_boxes, 5) containing pairwise
            position and time difference features 
        """
        track_coords = torchvision.ops.box_convert(
            track_coords, "xyxy", "cxcywh"
        )
        current_coords = torchvision.ops.box_convert(
            current_coords, "xyxy", "cxcywh"
        )

        (x_track, y_track, w_track, h_track) = track_coords.repeat(
            len(current_coords), 1, 1
        ).permute(2, 1, 0)

        (x_box, y_box, w_box, h_box) = current_coords.repeat(
            len(track_coords), 1, 1
        ).permute(2, 0, 1)

        x_norm = 2 * (x_box - x_track) / (h_box + h_track)
        y_norm = 2 * (y_box - y_track) / (h_box + h_track)
        h_norm = torch.log(h_track / h_box)
        w_norm = torch.log(w_track / w_box)
        t_norm = curr_t.unsqueeze(0) - track_t.unsqueeze(1)

        edge_feats = torch.stack(
            [x_norm, y_norm, w_norm, h_norm, t_norm], dim=-1
        )
        return edge_feats

    def forward(
        self,
        track_features,
        current_features,
        track_boxes,
        current_boxes,
        track_time,
        current_time,
        track_masks=None,
        current_masks=None,
    ):
        """
        Args:
            track_features: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_features: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_boxes: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_boxes: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)
                          
            track_time: track's timestamps, torch.Tensor with with shape (num_tracks, )
            current_time: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)
            
        Returns:
            train()
                classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                                containing at entry (step, i, j) the unnormalized probability that track i and 
                                detection j are a match, according to the classifier at the given neural message passing step
                [num_gnn_layers, num_track, num_boxes]

            eval()
                [num_track, num_boxes]
        """
        dist_reid = self.distance_fn(track_features, current_features)
        # Get initial edge embeddings to
        pos_edge_feats = self.compute_edge_feats(
            track_boxes, current_boxes, track_time, current_time
        )  # [num_tracks, num_boxes, 5]
        edge_feats = torch.cat(
            [pos_edge_feats, dist_reid.unsqueeze(-1)], dim=-1
        )  # [num_tracks, num_boxes, 6]   6=edge_in_dim
        fixed_edge_embeds = self.edge_in_mlp(
            edge_feats
        )  # [num_tracks, num_boxes, edge_dim]

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        track_embeds = F.relu(
            self.linear(track_features)
        )  # [num_tracks, 512] -> [num_tracks, node_dim]
        curr_embeds = F.relu(
            self.linear(current_features)
        )  # [num_tracks, 512] -> [num_tracks, node_dim]

        all_steps_edge_logits = []
        for i in range(self.num_steps):
            if i == 0:
                updatable_edge_embeds = fixed_edge_embeds.clone()

            edge_embeds = torch.cat(
                [updatable_edge_embeds, fixed_edge_embeds], dim=-1
            )  # [num_tracks, num_boxes, 2 * edge_dim]
            (
                updatable_edge_embeds,  # [num_tracks, num_boxes, edge_dim]
                track_embeds,  # [num_tracks, num_boxes, node_dim]
                curr_embeds,
            ) = self.graph_net(
                edge_embeds=edge_embeds,
                nodes_a_embeds=track_embeds,
                nodes_b_embeds=curr_embeds,
            )

            edge_logits = self.classifier(updatable_edge_embeds).squeeze(
                -1
            )  # [num_tracks, num_boxes]
            all_steps_edge_logits.append(edge_logits)

        if self.training:
            return torch.stack(
                all_steps_edge_logits
            )  # [num_steps, num_tracks, num_boxes]
        else:
            # distance is opposite of probability
            return 1 - torch.sigmoid(all_steps_edge_logits[-1])

