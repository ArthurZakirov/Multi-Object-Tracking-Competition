import collections

import numpy as np
import scipy

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import motmetrics as mm

mm.lap.default_solver = "lap"
import src.market.metrics as metrics
from src.utils.torch_utils import run_model_on_list
from src.tracker.utils import get_crop_from_boxes

_UNMATCHED_COST = 255


class Tracker:
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detect):
        self.obj_detect = obj_detect

        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.mot_accum = None

    def reset(self, hard=True):
        self.tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(new_boxes[i], new_scores[i], self.track_num + i)
            )
        self.track_num += num_new

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
        return box

    def update_results(self):
        """
        returns dictionary of the following structure
        
        results[track.id][im_index] = np.concatenate([box, score])

        """
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate(
                [t.box.cpu().numpy(), np.array([t.score])]
            )

        self.im_index += 1

    def data_association(self, boxes, scores):
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""

        # object detection
        # boxes, scores = self.obj_detect.detect(frame['img'])
        boxes, scores = frame["det"]["boxes"], frame["det"]["scores"]

        self.data_association(boxes, scores)
        self.update_results()

    def get_results(self):
        return self.results


class ReIDTracker(Tracker):
    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(
                    new_boxes[i],
                    new_scores[i],
                    self.track_num + i,
                    new_features[i],
                )
            )
        self.track_num += num_new

    def reset(self, hard=True):
        self.tracks = []
        # self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def data_association(self, boxes, scores, features):
        raise NotImplementedError

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
        boxes = frame["det"]["boxes"]
        scores = frame["det"]["scores"]
        reid_feats = frame["det"]["reid"].cpu()
        self.data_association(boxes, scores, reid_feats)

        # results
        self.update_results()

    def compute_distance_matrix(
        self,
        track_features,
        pred_features,
        track_boxes,
        boxes,
        metric_fn,
        alpha=0.0,
    ):
        UNMATCHED_COST = 255.0

        # Build cost matrix.
        distance = mm.distances.iou_matrix(
            track_boxes.numpy(), boxes.numpy(), max_iou=0.5
        )

        appearance_distance = metrics.compute_distance_matrix(
            track_features, pred_features, metric_fn=metric_fn
        )
        appearance_distance = appearance_distance.numpy() * 0.5
        # return appearance_distance

        assert np.alltrue(appearance_distance >= -0.1)
        assert np.alltrue(appearance_distance <= 1.1)

        combined_costs = alpha * distance + (1 - alpha) * appearance_distance

        # Set all unmatched costs to _UNMATCHED_COST.
        distance = np.where(np.isnan(distance), UNMATCHED_COST, combined_costs)

        distance = np.where(appearance_distance > 0.1, UNMATCHED_COST, distance)

        return distance


class MyTracker(ReIDTracker):
    def __init__(
        self,
        assign_model,
        reid_model,
        obj_detect,
        distance_threshold,
        unmatched_cost,
        patience,
    ):

        super().__init__(obj_detect=obj_detect)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reid_model = reid_model
        self.assign_model = assign_model

        self.distance_threshold = distance_threshold
        self.unmatched_cost = unmatched_cost
        self.patience = patience

    @classmethod
    def from_config(cls, hyperparams):
        obj_detect = (
            None
            if hyperparams["obj_detect"] is None
            else torch.load(hyperparams["obj_detect"])
        )
        reid_model = (
            None
            if hyperparams["reid_model"] is None
            else torch.load(hyperparams["reid_model"])
        )
        assign_model = (
            None
            if hyperparams["assign_model"] is None
            else torch.load(hyperparams["assign_model"])
        )

        hyperparams.pop("obj_detect", None)
        hyperparams.pop("reid_model", None)
        hyperparams.pop("assign_model", None)

        tracker = cls.__new__(cls)
        tracker.__init__(
            **hyperparams,
            assign_model=assign_model,
            reid_model=reid_model,
            obj_detect=obj_detect
        )
        return tracker

    @property
    def track_boxes(self):
        return torch.stack([t.box for t in self.tracks], axis=0)

    @property
    def track_ids(self):
        return torch.stack([t.id for t in self.tracks], axis=0)

    @property
    def track_features(self):
        return torch.stack([t.get_feature() for t in self.tracks], axis=0)

    @property
    def track_masks(self):
        return torch.stack([t.get_mask() for t in self.tracks], axis=0)

    @property
    def track_time(self):
        return torch.Tensor(
            [self.im_index - t.inactive - 1 for t in self.tracks]
        )

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob containing the image information.
		"""
        # take detection from databasis if availablet
        if "boxes" in frame.keys():
            detection = frame
        else:
            with torch.no_grad():
                self.obj_detect.eval()
                self.obj_detect.to(self.device)
                img = frame["img"].to(self.device)
                detection = self.obj_detect(img)[0]

        scores = detection["scores"].cpu()
        boxes = detection["boxes"].cpu()
        masks = (
            torch.nan * torch.ones((len(scores)))
            if not "masks" in detection.keys()
            else detection["masks"].cpu()
        )

        # take reid features from databasis if available
        if "reid" in frame.keys():
            pred_features = frame["reid"].cpu()
        elif not self.reid_model is None:
            crops = self.get_crop_from_boxes(boxes, frame)
            self.reid_model.eval()
            with torch.no_grad():
                pred_features = run_model_on_list(
                    model=self.reid_model,
                    input_list=crops,
                    device=self.device,
                    concat=True,
                )
        else:
            pred_features = torch.nan * torch.ones((len(scores)))

        # print("\n\nboxes", boxes.shape)
        # print("scores", scores.shape)
        # print("pred_features", pred_features.shape)
        self.data_association(boxes, scores, pred_features, masks)
        self.update_results()

    def update_results(self):
        """Only store boxes for tracks that are active"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            if t.inactive == 0:
                self.results[t.id][self.im_index] = np.concatenate(
                    [t.box.cpu().numpy(), np.array([t.score])]
                )
        self.im_index += 1

    def add(self, new_boxes, new_scores, new_features, new_masks):
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(
                    box=new_boxes[i],
                    score=new_scores[i],
                    track_id=self.track_num + i,
                    feature=new_features[i],
                    inactive=0,
                    mask=new_masks[i],
                )
            )
        self.track_num += num_new

    def ùpdate_tracks(
        self,
        matched_track_idx,
        unmatched_track_idx,
        matched_box_idx,
        unmatched_box_idx,
        boxes,
        scores,
        pred_features,
        masks,
    ):

        # update matched tracks
        for track_idx, box_idx in zip(matched_track_idx, matched_box_idx):
            self.tracks[track_idx].box = boxes[box_idx]
            self.tracks[track_idx].add_feature(pred_features[box_idx])

        # set matched tracks to "active"
        for track_idx in matched_track_idx:
            self.tracks[track_idx].inactive = 0

        # set unmatched tracks to "inactive"
        for track_idx in unmatched_track_idx:
            self.tracks[track_idx].inactive += 1

        # remove long inactive tracks
        remove_track_ids = []
        for track_idx in unmatched_track_idx:
            self.tracks[track_idx].inactive += 1
            if self.tracks[track_idx].inactive > self.patience:
                remove_track_ids.append(self.tracks[track_idx].id)
        self.tracks = [t for t in self.tracks if not t.id in remove_track_ids]

        # add new
        new_boxes = [boxes[idx] for idx in unmatched_box_idx]
        new_scores = [scores[idx] for idx in unmatched_box_idx]
        new_features = [pred_features[idx] for idx in unmatched_box_idx]
        new_masks = [masks[idx] for idx in unmatched_box_idx]
        self.add(new_boxes, new_scores, new_features, new_masks)

    def data_association(self, boxes, scores, pred_features, masks):
        """
    This method performs the management of the current tracks

    Arguments
    ---------
    boxes [N, 4]
      - detector output boxes in the xyxy format

    scores [N]
      - detector output pedestrian probability
    """

        if len(self.tracks) == 0:
            self.add(
                new_boxes=boxes,
                new_scores=scores,
                new_features=pred_features,
                new_masks=masks,
            )

        else:
            self.assign_model.eval()
            with torch.no_grad():
                distance_matrix = self.assign_model(
                    track_features=self.track_features,
                    current_features=pred_features,
                    track_boxes=self.track_boxes,
                    current_boxes=boxes,
                    track_time=self.track_time,
                    current_time=self.im_index * torch.ones_like(scores),
                    track_masks=self.track_masks,
                    current_masks=masks,
                )

            distance_matrix = np.where(
                distance_matrix > self.distance_threshold,
                self.unmatched_cost,
                distance_matrix,
            )

            (
                matched_track_idx,
                unmatched_track_idx,
                matched_box_idx,
                unmatched_box_idx,
            ) = hungarian_matching(
                distance_matrix=distance_matrix,
                unmatched_cost=self.unmatched_cost,
            )

            self.ùpdate_tracks(
                matched_track_idx,
                unmatched_track_idx,
                matched_box_idx,
                unmatched_box_idx,
                boxes,
                scores,
                pred_features,
                masks,
            )


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(
        self,
        box,
        score,
        track_id,
        feature=None,
        inactive=0,
        mask=None,
        max_features_num=10,
    ):
        self.id = track_id
        self.box = box
        self.score = score
        self.feature = collections.deque([feature])
        self.mask = mask
        self.inactive = inactive
        self.max_features_num = max_features_num

    def add_feature(self, feature):
        """Adds new appearance features to the object."""
        self.feature.append(feature)
        if len(self.feature) > self.max_features_num:
            self.feature.popleft()

    def get_feature(self):
        if len(self.feature) > 1:
            feature = torch.stack(list(self.feature), dim=0)
        else:
            feature = self.feature[0].unsqueeze(0)
        return feature[-1]

    def get_mask(self):
        return self.mask


def hungarian_matching(distance_matrix, unmatched_cost):
    row_idx, col_idx = linear_sum_assignment(distance_matrix)
    costs = distance_matrix[row_idx, col_idx]
    select_matched = np.where(costs != unmatched_cost)[0]

    total_track_idx = range(distance_matrix.shape[0])
    total_box_idx = range(distance_matrix.shape[1])

    matched_track_idx = row_idx[select_matched].tolist()
    matched_box_idx = col_idx[select_matched].tolist()

    unmatched_track_idx = list(set(total_track_idx) - set(matched_track_idx))
    unmatched_box_idx = list(set(total_box_idx) - set(matched_box_idx))

    return (
        matched_track_idx,
        unmatched_track_idx,
        matched_box_idx,
        unmatched_box_idx,
    )

