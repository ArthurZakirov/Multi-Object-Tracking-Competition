import collections
import json
import numpy as np
import scipy

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import motmetrics as mm

from src.motion_prediction.model import KalmanPredictor

mm.lap.default_solver = "lap"
import src.market.metrics as metrics
from src.utils.torch_utils import run_model_on_list
from src.tracker.utils import get_crop_from_boxes
from src.motion_prediction.kalman import FullBoxFilter

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


class MyTracker:
    # TODO : don't worry about computational efficiency yet,
    # you can later define params such as frequency of controlling long inactive tracks if they reappeared
    # instead of doing it on every timestep

    def __init__(
        self,
        assign_model,
        reid_model,
        obj_detect,
        distance_threshold,
        unmatched_cost,
        patience,
        long_inactive_thresh,
        submit_track_status,
        short_motion_predictor,
        long_motion_predictor,
        **kwargs,
    ):

        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obj_detect = obj_detect
        self.reid_model = reid_model
        self.assign_model = assign_model
        self.short_motion_predictor = short_motion_predictor
        self.long_motion_predictor = long_motion_predictor

        self.distance_threshold = distance_threshold
        self.unmatched_cost = unmatched_cost
        self.patience = patience
        self._submit_track_status = submit_track_status

        self.scene = Scene(long_inactive_thresh)
        self.results = {}

    def reset(self):
        self.scene.reset()
        self.results = {}

    def get_results(self):
        return self.results

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

        if hyperparams["short_motion_predictor"] is None:
            short_motion_predictor = None
        elif ".pth" in hyperparams["short_motion_predictor"]:
            short_motion_predictor = torch.load(
                hyperparams["short_motion_predictor"]
            )
        elif "kalman" in hyperparams["short_motion_predictor"]:
            with open(hyperparams["short_motion_predictor"], "r") as f:
                predictor_config = json.load(f)
            short_motion_predictor = KalmanPredictor.from_config(
                predictor_config
            )

        if hyperparams["long_motion_predictor"] is None:
            long_motion_predictor = None
        elif ".pth" in hyperparams["long_motion_predictor"]:
            long_motion_predictor = torch.load(
                hyperparams["long_motion_predictor"]
            )
        elif "kalman" in hyperparams["long_motion_predictor"]:
            with open(hyperparams["long_motion_predictor"], "r") as f:
                predictor_config = json.load(f)
            long_motion_predictor = KalmanPredictor.from_config(
                predictor_config
            )

        hyperparams.pop("obj_detect", None)
        hyperparams.pop("reid_model", None)
        hyperparams.pop("assign_model", None)
        hyperparams.pop("short_motion_predictor", None)
        hyperparams.pop("long_motion_predictor", None)

        tracker = cls.__new__(cls)
        tracker.__init__(
            **hyperparams,
            assign_model=assign_model,
            reid_model=reid_model,
            obj_detect=obj_detect,
            short_motion_predictor=short_motion_predictor,
            long_motion_predictor=long_motion_predictor,
        )
        return tracker

    def step(self, frame):
        boxes, scores, pred_features, masks = self._compute_detection(frame)
        self._data_association(boxes, scores, pred_features, masks)
        self._update_results()

    def _compute_detection(self, frame):
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
        return boxes, scores, pred_features, masks

    def _predict_track_boxes(self):
        s = self.scene

        active_ids = s.track_ids("active")
        long_inactive_ids = s.track_ids("long_inactive")
        short_inactive_ids = s.track_ids("short_inactive")
        with torch.no_grad():
            if not self.short_motion_predictor is None and active_ids:
                self.short_motion_predictor.eval()
                active_trajs = s.track_trajectories("active")
                active_pred_trajs = self.short_motion_predictor(
                    trajs=active_trajs,
                    future_lens=[1 for _ in range(len(active_trajs))],
                )
                active_pred_boxes = [traj[-1] for traj in active_pred_trajs]
            else:
                active_pred_boxes = s.track_boxes("active")

            if not self.short_motion_predictor is None and short_inactive_ids:
                short_inactive_trajs = s.track_trajectories("short_inactive")
                short_inactive_counts = s.track_inactive_counts(
                    "short_inactive"
                )
                short_inactive_pred_trajs = self.short_motion_predictor(
                    trajs=short_inactive_trajs,
                    future_lens=[count + 1 for count in short_inactive_counts],
                )
                short_inactive_pred_boxes = [
                    traj[-1] for traj in short_inactive_pred_trajs
                ]

            else:
                short_inactive_pred_boxes = s.track_boxes("short_inactive")

            if not self.long_motion_predictor is None and long_inactive_ids:
                long_inactive_trajs = s.track_trajectories("long_inactive")
                long_inactive_counts = s.track_inactive_counts("long_inactive")
                self.long_motion_predictor.eval()
                long_inactive_pred_trajs = self.long_motion_predictor(
                    trajs=long_inactive_trajs,
                    future_lens=[count + 1 for count in long_inactive_counts],
                )
                long_inactive_pred_boxes = [
                    traj[-1] for traj in long_inactive_pred_trajs
                ]
            else:
                long_inactive_pred_boxes = s.track_boxes("long_inactive")

        pred_box_dict = {}
        pred_box_dict.update(dict(zip(active_ids, active_pred_boxes)))
        pred_box_dict.update(
            dict(zip(long_inactive_ids, long_inactive_pred_boxes))
        )
        pred_box_dict.update(
            dict(zip(short_inactive_ids, short_inactive_pred_boxes))
        )

        pred_box_tensor = torch.stack(
            list(dict(sorted(pred_box_dict.items())).values()), dim=0
        )
        return pred_box_tensor

    def _update_results(self):
        """Only store boxes for tracks that are active"""
        s = self.scene
        submit_tracks = []
        for status in self._submit_track_status:
            submit_tracks.extend(s.get_tracks(status))

        for t in submit_tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}

            self.results[t.id][s.im_index] = np.concatenate(
                [t.get_box().cpu().numpy(), np.array([t.score])]
            )
        s.update_im_index()

    def _ùpdate_tracks(
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
        s = self.scene
        tracks = s.tracks
        # update matched tracks
        for track_idx, box_idx in zip(matched_track_idx, matched_box_idx):
            t = tracks[track_idx]
            t.add_box(boxes[box_idx])
            t.add_feature(pred_features[box_idx])

        # set matched tracks to "active"
        for track_idx in matched_track_idx:
            t = tracks[track_idx]
            t.inactive = 0

        # set unmatched tracks to "inactive"
        for track_idx in unmatched_track_idx:
            t = tracks[track_idx]
            t.inactive += 1

        # increase inactive count, and remove long inactive tracks
        remove_track_ids = []
        for track_idx in unmatched_track_idx:
            t = tracks[track_idx]
            t.inactive += 1
            if t.inactive > self.patience:
                remove_track_ids.append(t.id)
        tracks = [t for t in tracks if not t.id in remove_track_ids]

        # add new
        new_boxes = [boxes[idx] for idx in unmatched_box_idx]
        new_scores = [scores[idx] for idx in unmatched_box_idx]
        new_features = [pred_features[idx] for idx in unmatched_box_idx]
        new_masks = [masks[idx] for idx in unmatched_box_idx]

        s.add(new_boxes, new_scores, new_features, new_masks)

    def _data_association(self, boxes, scores, pred_features, masks):
        """
        This method performs the management of the current tracks

        Arguments
        ---------
        boxes [N, 4]
        - detector output boxes in the xyxy format

        scores [N]
        - detector output pedestrian probability
        """
        s = self.scene
        if len(s.tracks) == 0:
            s.add(
                new_boxes=boxes,
                new_scores=scores,
                new_features=pred_features,
                new_masks=masks,
            )
        else:
            track_boxes = self._predict_track_boxes()
            self.assign_model.eval()
            with torch.no_grad():
                distance_matrix = self.assign_model(
                    track_features=s.track_features("all"),
                    current_features=pred_features,
                    track_boxes=track_boxes,
                    current_boxes=boxes,
                    track_time=s.track_time("all"),
                    current_time=s.im_index * torch.ones_like(scores),
                    track_masks=s.track_masks("all"),
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

            self._ùpdate_tracks(
                matched_track_idx,
                unmatched_track_idx,
                matched_box_idx,
                unmatched_box_idx,
                boxes,
                scores,
                pred_features,
                masks,
            )


class Scene:
    def __init__(self, long_inactive_thresh):
        self.tracks = []
        self.im_index = 0
        self.long_inactive_thresh = long_inactive_thresh

    def reset(self):
        self.tracks = []
        self.im_index = 0

    def update_im_index(self):
        self.im_index += 1

    def get_tracks(self, status="all"):
        all_tracks = self.tracks
        active_tracks = [t for t in self.tracks if t.inactive == 0]
        long_inactive_tracks = [
            t for t in self.tracks if t.inactive > self.long_inactive_thresh
        ]
        short_inactive_tracks = [
            t
            for t in self.tracks
            if (t.inactive > 0 and t.inactive <= self.long_inactive_thresh)
        ]
        if status == "all":
            return all_tracks
        elif status == "active":
            return active_tracks
        elif status == "long_inactive":
            return long_inactive_tracks
        elif status == "short_inactive":
            return short_inactive_tracks

    def track_boxes(self, status="all"):
        return [t.get_box() for t in self.get_tracks(status)]

    def track_trajectories(self, status="all"):
        return [t.get_trajectory() for t in self.get_tracks(status)]

    def track_inactive_counts(self, status="all"):
        return [t.inactive for t in self.get_tracks(status)]

    def track_ids(self, status="all"):
        return [t.id for t in self.get_tracks(status)]

    def track_features(self, status="all"):
        return torch.stack(
            [t.get_feature() for t in self.get_tracks(status)], axis=0
        )

    def track_masks(self, status="all"):
        return torch.stack(
            [t.get_mask() for t in self.get_tracks(status)], axis=0
        )

    def track_time(self, status="all"):
        return torch.Tensor(
            [self.im_index - t.inactive - 1 for t in self.get_tracks(status)]
        )

    def add(self, new_boxes, new_scores, new_features, new_masks):
        num_new = len(new_boxes)
        for i in range(num_new):
            new_track = Track(
                box=new_boxes[i],
                score=new_scores[i],
                track_id=len(self.tracks) + 1,
                feature=new_features[i],
                inactive=0,
                mask=new_masks[i],
            )
            self.tracks.append(new_track)


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
        self.boxes = collections.deque([box])
        self.score = score
        self.features = collections.deque([feature])
        self.masks = collections.deque([mask])
        self.inactive = inactive
        self.max_features_num = max_features_num

    def add_box(self, box):
        self.boxes.append(box)
        if len(self.boxes) > self.max_features_num:
            self.boxes.popleft()

    def get_box(self):
        return self.boxes[-1]

    def get_trajectory(self):
        return torch.stack(list(self.boxes), dim=0)

    def add_mask(self, mask):
        self.masks.append(mask)
        if len(self.masks) > self.max_features_num:
            self.masks.popleft()

    def get_mask(self):
        return self.masks[-1]

    def add_feature(self, feature):
        """Adds new appearance features to the object."""
        self.features.append(feature)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def get_feature(self):
        if len(self.features) > 1:
            feature = torch.stack(list(self.features), dim=0)
        else:
            feature = self.features[0].unsqueeze(0)
        return feature[-1]


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

