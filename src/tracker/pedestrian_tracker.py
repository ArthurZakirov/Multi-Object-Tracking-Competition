import collections
import os
import json
import numpy as np
import scipy

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import motmetrics as mm

from src.motion_prediction.model import KalmanPredictor, FullFilter
from src.tracker.tracker import obj_is_occluded
from src.tracker.tracker import hungarian_matching

mm.lap.default_solver = "lap"
import src.market.metrics as metrics
from src.utils.torch_utils import run_model_on_list
from src.tracker.utils import prepare_crops_for_reid
import torchvision
from src.motion_prediction.kalman import SORTKalmanFilter, obj_is_moving

_UNMATCHED_COST = 255


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(
        self,
        box,
        score,
        mask=None,
        feature=None,
        track_id=None,
        inactive=0,
        max_features_num=100,
    ):
        self.boxes = collections.deque([box])
        self.score = score
        self.features = collections.deque([feature])
        self.masks = collections.deque([mask])

        self.id = track_id
        self.inactive = inactive
        self.max_features_num = max_features_num

    def add_box(self, box):
        self.boxes.append(box)
        if len(self.boxes) > self.max_features_num:
            self.boxes.popleft()

    def get_result(self):
        box = self.boxes[-1].cpu().numpy().squeeze()
        score = np.array([self.score])
        return np.concatenate([box, score])

    def get_box(self):
        return self.boxes[-1].squeeze()

    def get_trajectory(self):
        return torch.stack(list(self.boxes), dim=0)

    def set_trajectory(self, traj):
        self.boxes = collections.deque(traj.flatten().chunk(chunks=len(traj)))

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


class KalmanTrack(Track):
    def __init__(self, box, **kwargs):
        super().__init__(box=box, **kwargs)
        self.kalman = SORTKalmanFilter()
        self.kalman.update(box)
        self.preds = box.unsqueeze(0)

    def get_result(self):
        box = self.preds[-1].cpu().numpy()
        score = np.array([self.score])
        return np.concatenate([box, score])

    def step_kalman(self):
        pred = self.kalman.predict()
        self.preds = torch.cat([self.preds, pred.unsqueeze(0)], dim=0)

    def get_box(self):
        return self.preds[-1]

    def add_box(self, box):
        super().add_box(box)
        self.kalman.update(box)


class DistractorAwareTrack(KalmanTrack):
    def __init__(
        self, distractor=False, not_moving=0, not_moving_thresh=10, **kwargs
    ):
        super().__init__(**kwargs)
        self._distractor = distractor
        self.not_moving_thresh = not_moving_thresh
        self.not_moving = not_moving

    @property
    def distractor(self):
        return self._distractor or self.not_moving > self.not_moving_thresh


class BoxCorrectiveTrack(KalmanTrack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.submission_kalman = None
        self.misjudged_mask = torch.tensor([], dtype=torch.bool)

    def set_judged(self):
        self.misjudged_mask = torch.cat(
            [self.misjudged_mask, torch.tensor([False], dtype=torch.bool)]
        )

    def set_misjudged(self):
        self.misjudged_mask = torch.cat(
            [self.misjudged_mask, torch.tensor([True], dtype=torch.bool)]
        )

    # def get_misjudged_box_pred(self):
    #     first_det_misjudged = self.misjudged_mask[0]
    #     if first_det_misjudged:
    #         return self.preds[-1]
    #     else:
    #         last_judged_idx = torch.where(self.misjudged_mask)[0][0] - 1
    #     preds = []
    #     self.submission_kalman = SORTKalmanFilter()
    #     for timestep in range(len(self.misjudged_mask)):
    #         box = self.boxes[timestep]
    #         if timestep <= last_judged_idx:
    #             self.submission_kalman.update(box)
    #         pred = self.submission_kalman.predict()
    #     preds.append(pred)
    #     return preds[-1]

    def get_result(self):
        result = self.preds[-1]
        if self.misjudged_mask.any() and not self.misjudged_mask.all():
            if self.misjudged_mask[-1]:
                last_judge = torch.where(~self.misjudged_mask)[0][-1]
                result = self.preds[last_judge]
        box = result.cpu().numpy()
        score = np.array([self.score])
        return np.concatenate([box, score])


class TrackerBasis:
    def __init__(self, long_inactive_thresh=0, track_cls=Track):
        self.track_cls = track_cls
        self.tracks = []
        self.im_index = 0
        self.num_tracks = 0
        self.long_inactive_thresh = long_inactive_thresh
        self.results = {}

    def get_results(self):
        return self.results

    def reset(self):
        self.tracks = []
        self.results = {}
        self.im_index = 0

    def update_im_index(self):
        self.im_index += 1

    @property
    def active_tracks(self):
        return [t for t in self.tracks if t.inactive == 0]

    @property
    def short_inactive_tracks(self):
        return [
            t
            for t in self.tracks
            if (t.inactive > 0 and t.inactive <= self.long_inactive_thresh)
        ]

    @property
    def long_inactive_tracks(self):
        return [
            t for t in self.tracks if t.inactive > self.long_inactive_thresh
        ]

    def get_tracks(self, status="all"):
        ids = [t.id for t in self.tracks]
        assert len(ids) == len(set(ids))
        if status == "all":
            return self.tracks
        elif status == "active":
            return self.active_tracks
        elif status == "long_inactive":
            return self.long_inactive_tracks
        elif status == "short_inactive":
            return self.short_inactive_tracks

    def prev_track_boxes(self, status="all"):
        return torch.stack(
            [t.boxes[-2] for t in self.get_tracks(status)], dim=0
        )

    def track_boxes(self, status="all"):
        return torch.stack(
            [t.get_box() for t in self.get_tracks(status)], dim=0
        )

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
            new_track = self.track_cls(
                box=new_boxes[i],
                score=new_scores[i],
                track_id=self.num_tracks + 1,
                feature=new_features[i],
                inactive=0,
                mask=new_masks[i],
            )
            self.num_tracks += 1
            self.tracks.append(new_track)

    def step(self, frame):
        boxes, scores, masks = self._get_detection(frame)
        reid_features = self._get_reid_features(frame, boxes, masks)
        if len(self.tracks) == 0:
            self._initialize_tracks(boxes, scores, reid_features, masks)
        else:
            (
                matched_track_idx,
                unmatched_track_idx,
                matched_box_idx,
                unmatched_box_idx,
            ) = self._data_association(boxes, scores, reid_features, masks)
            self._update_tracks(
                matched_track_idx,
                unmatched_track_idx,
                matched_box_idx,
                unmatched_box_idx,
                boxes,
                scores,
                reid_features,
                masks,
            )
        self._update_results()

    def _get_detection(self):
        raise NotImplementedError()

    def _get_reid_features(self):
        raise NotImplementedError()

    def _data_association(self):
        raise NotImplementedError()

    def _initialize_tracks(self):
        raise NotImplementedError()

    def _update_tracks(
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
        self._manage_matches(
            matched_track_idx, matched_box_idx, boxes, pred_features
        )
        self._manage_unmatched_tracks(unmatched_track_idx)
        self._manage_unmatched_boxes(
            unmatched_box_idx, boxes, scores, pred_features, masks
        )

    def _manage_matches(self):
        raise NotImplementedError()

    def _manage_unmatched_tracks(self):
        raise NotImplementedError()

    def _manage_unmatched_boxes(self):
        raise NotImplementedError()

    def _update_results(self):
        submit_tracks = []
        for status in self._submit_track_status:
            submit_tracks.extend(self.get_tracks(status))
        for t in submit_tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            t_track_result = t.get_result()
            self.results[t.id][self.im_index] = t_track_result
        self.update_im_index()


class Tracker(TrackerBasis):
    def __init__(
        self,
        track_cls=Track,
        assign_model=None,
        reid_model=None,
        obj_detect=None,
        distance_threshold=0.5,
        unmatched_cost=255,
        long_inactive_thresh=0,
        submit_track_status=["active"],
        **kwargs,
    ):

        super().__init__(
            track_cls=track_cls, long_inactive_thresh=long_inactive_thresh,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obj_detect = obj_detect
        self.reid_model = reid_model
        self.assign_model = assign_model

        self.distance_threshold = distance_threshold
        self.unmatched_cost = unmatched_cost
        self._submit_track_status = submit_track_status

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
        if not hyperparams["reid_model"] is None:
            reid_model.input_is_masked = json.load(
                open(
                    os.path.join(
                        os.path.dirname(hyperparams["reid_model"]),
                        "model_config.json",
                    ),
                    "r",
                )
            )["input_is_masked"]

        assign_model = (
            None
            if hyperparams["assign_model"] is None
            else torch.load(hyperparams["assign_model"])
        )

        byte_assign_model = (
            None
            if hyperparams["byte_assign_model"] is None
            else torch.load(hyperparams["byte_assign_model"])
        )

        hyperparams.pop("obj_detect", None)
        hyperparams.pop("reid_model", None)
        hyperparams.pop("assign_model", None)
        hyperparams.pop("byte_assign_model", None)

        tracker = cls.__new__(cls)
        tracker.__init__(
            assign_model=assign_model,
            byte_assign_model=byte_assign_model,
            reid_model=reid_model,
            obj_detect=obj_detect,
            **hyperparams,
        )
        return tracker

    def _get_detection(self, frame):
        # take detection from databasis if available
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
        return boxes, scores, masks

    def _get_reid_features(self, frame, boxes, masks):
        # take reid features from databasis if available
        if "reid" in frame.keys():
            pred_features = frame["reid"].cpu()
        elif not self.reid_model is None:
            self.reid_model.eval()
            reid_masks = masks if self.reid_model.input_is_masked else None
            crops = prepare_crops_for_reid(
                image=frame["img"], boxes=boxes, masks=reid_masks
            )
            with torch.no_grad():
                pred_features = run_model_on_list(
                    model=self.reid_model,
                    input_list=crops,
                    device=self.device,
                    concat=True,
                )
        else:
            pred_features = torch.nan * torch.ones((len(boxes)))
        return pred_features

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
        self.assign_model.eval()
        confident = scores > 0.5
        with torch.no_grad():
            distance_matrix = self.assign_model(
                track_features=self.track_features("all"),
                current_features=pred_features[confident],
                track_boxes=self.track_boxes("all"),
                current_boxes=boxes[confident],
                track_time=self.track_time("all"),
                current_time=self.im_index * torch.ones(len(boxes[confident])),
                track_masks=self.track_masks("all"),
                current_masks=masks[confident],
            )

        (
            matched_track_idx,
            unmatched_track_idx,
            matched_box_idx,
            unmatched_box_idx,
        ) = hungarian_matching(
            distance_matrix=distance_matrix,
            unmatched_cost=self.unmatched_cost,
            distance_threshold=self.distance_threshold,
        )
        return (
            matched_track_idx,
            unmatched_track_idx,
            matched_box_idx,
            unmatched_box_idx,
        )

    def _initialize_tracks(self, boxes, scores, reid_features, masks):
        self.add(boxes, scores, reid_features, masks)

    def _manage_matches(
        self, matched_track_idx, matched_box_idx, boxes, pred_features
    ):
        for track_idx, box_idx in zip(matched_track_idx, matched_box_idx):
            t = self.tracks[track_idx]
            t.add_box(boxes[box_idx])
            t.add_feature(pred_features[box_idx])

    def _manage_unmatched_tracks(self, unmatched_track_idx):
        remove_track_ids = []
        for track_idx in unmatched_track_idx:
            t = self.tracks[track_idx]
            remove_track_ids.append(t.id)
        self.tracks = [t for t in self.tracks if not t.id in remove_track_ids]

    def _manage_unmatched_boxes(
        self, unmatched_box_idx, boxes, scores, pred_features, masks
    ):
        new_boxes = []
        new_scores = []
        new_features = []
        new_masks = []
        for idx in unmatched_box_idx:
            new_boxes.append(boxes[idx])
            new_scores.append(scores[idx])
            new_features.append(pred_features[idx])
            new_masks.append(masks[idx])
        self.add(new_boxes, new_scores, new_features, new_masks)


class KalmanTracker(Tracker):
    def __init__(self, track_cls=KalmanTrack, **kwargs):
        super().__init__(track_cls=track_cls, **kwargs)

    def _assert_kalman_steps(self):
        for t in self.tracks:
            len(t.boxes) == len(t.preds)

    def _step_kalman(self):
        for t in self.tracks:
            t.step_kalman()

    def _data_association(self, boxes, scores, reid_features, masks):
        self._step_kalman()
        self._assert_kalman_steps()
        return super()._data_association(boxes, scores, reid_features, masks)


class PatientTracker(KalmanTracker):
    def __init__(self, patience, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience

    def _manage_matches(
        self, matched_track_idx, matched_box_idx, boxes, pred_features
    ):
        for track_idx, box_idx in zip(matched_track_idx, matched_box_idx):
            t = self.tracks[track_idx]
            t.add_box(boxes[box_idx])
            t.add_feature(pred_features[box_idx])
            t.inactive = 0

    def _manage_unmatched_tracks(self, unmatched_track_idx):
        remove_track_ids = []
        for track_idx in unmatched_track_idx:
            t = self.tracks[track_idx]
            t.inactive += 1
            if t.inactive > self.patience:
                remove_track_ids.append(t.id)
        self.tracks = [t for t in self.tracks if not t.id in remove_track_ids]

    def _manage_unmatched_boxes(
        self, unmatched_box_idx, boxes, scores, pred_features, masks
    ):
        new_boxes = []
        new_scores = []
        new_features = []
        new_masks = []
        for idx in unmatched_box_idx:
            new_boxes.append(boxes[idx])
            new_scores.append(scores[idx])
            new_features.append(pred_features[idx])
            new_masks.append(masks[idx])
        self.add(new_boxes, new_scores, new_features, new_masks)


class BoxCorrectionTracker(PatientTracker):
    def __init__(self, track_cls=BoxCorrectiveTrack, **kwargs):
        super().__init__(track_cls=track_cls, **kwargs)

    def _manage_matches(
        self, matched_track_idx, matched_box_idx, boxes, pred_features
    ):
        super()._manage_matches(
            matched_track_idx, matched_box_idx, boxes, pred_features
        )
        misjudged_idxs = detect_misjudged_boxes(
            boxes=self.track_boxes("all")[matched_track_idx],
            next_boxes=boxes[matched_box_idx],
        )
        for idx in matched_track_idx:
            t = self.tracks[idx]
            if idx in misjudged_idxs:
                t.set_misjudged()
            else:
                t.set_judged()


class ByteTracker(PatientTracker):
    def __init__(self, byte_assign_model, **kwargs):
        super().__init__(**kwargs)
        self.byte_assign_model = byte_assign_model

    def _data_association(self, boxes, scores, pred_features, masks):
        (
            matched_track_idx,
            unmatched_track_idx,
            matched_box_idx,
            unmatched_box_idx,
        ) = super()._data_association(boxes, scores, pred_features, masks)
        confident = scores > 0.5
        byte_distance_matrix = self.byte_assign_model(
            track_features=self.track_features("all")[unmatched_track_idx],
            current_features=pred_features[~confident],
            track_boxes=self.track_boxes("all")[unmatched_track_idx],
            current_boxes=boxes[~confident],
            track_time=self.track_time("all")[unmatched_track_idx],
            current_time=self.im_index * torch.ones(len(boxes[~confident])),
            track_masks=self.track_masks("all")[unmatched_track_idx],
            current_masks=masks[~confident],
        )

        (
            local_matched_track_idx,
            local_unmatched_track_idx,
            local_matched_box_idx,
            local_unmatched_box_idx,
        ) = hungarian_matching(
            distance_matrix=byte_distance_matrix,
            unmatched_cost=self.unmatched_cost,
            distance_threshold=self.distance_threshold,
        )
        matched_track_idx.extend(
            np.array(unmatched_track_idx)[local_matched_track_idx].tolist()
        )
        unmatched_track_idx = np.array(unmatched_track_idx)[
            local_unmatched_track_idx
        ].tolist()
        matched_box_idx.extend(
            np.where(boxes[~confident])[0][local_matched_box_idx].tolist()
        )
        unmatched_box_idx = unmatched_box_idx

        return (
            matched_track_idx,
            unmatched_track_idx,
            matched_box_idx,
            unmatched_box_idx,
        )


class DistractorAwareTracker(PatientTracker):
    def __init__(
        self, track_cls=DistractorAwareTrack, not_moving_thresh=100, **kwargs
    ):
        super().__init__(track_cls=track_cls, **kwargs)
        self.not_moving_thresh = not_moving_thresh

    @property
    def active_tracks(self):
        return [t for t in self.tracks if t.inactive == 0 and not t.distractor]

    @property
    def short_inactive_tracks(self):
        return [
            t
            for t in self.tracks
            if (
                t.inactive > 0
                and t.inactive <= self.long_inactive_thresh
                and not t.distractor
            )
        ]

    def add(
        self,
        new_boxes,
        new_scores,
        new_features,
        new_masks,
        distractor_bool=None,
    ):
        num_new = len(new_boxes)
        if distractor_bool is None:
            distractor_bool = torch.zeros_like(new_scores, dtype=torch.bool)
        for i in range(num_new):
            new_track = self.track_cls(
                box=new_boxes[i],
                score=new_scores[i],
                track_id=self.num_tracks + 1,
                feature=new_features[i],
                inactive=0,
                mask=new_masks[i],
                not_moving_thresh=self.not_moving_thresh,
                distractor=distractor_bool[i],
            )
            self.num_tracks += 1
            self.tracks.append(new_track)

    def _initialize_tracks(self, boxes, scores, reid_features, masks):
        high_confident = scores > 0.5
        keep = high_confident
        confident_boxes = boxes[~high_confident]
        unconfident_boxes = boxes[high_confident]

        for idx, box in enumerate(unconfident_boxes):
            if obj_is_occluded(obj_box=box, other_boxes=confident_boxes,):
                keep[idx] = True

        self.add(boxes[keep], scores[keep], reid_features[keep], masks[keep])

    def _manage_matches(
        self, matched_track_idx, matched_box_idx, boxes, pred_features
    ):
        """remove long not moving tracks"""
        super()._manage_matches(
            matched_track_idx, matched_box_idx, boxes, pred_features
        )
        for track_idx in matched_track_idx:
            t = self.tracks[track_idx]
            if not obj_is_moving(t.get_trajectory()):
                t.not_moving += 1
            else:
                t.not_moving = 0

    def _manage_unmatched_boxes(
        self, unmatched_box_idx, boxes, scores, pred_features, masks
    ):
        """after first frame, only accept new objects, if they have been occluded in the previous frame"""
        distractor_mask = []
        for idx in unmatched_box_idx:
            distractor_status = not obj_is_occluded(
                obj_box=boxes[idx], other_boxes=self.prev_track_boxes("active"),
            )
            distractor_mask.append(distractor_status)
        distractor_mask = torch.tensor(distractor_mask)

        new_boxes = []
        new_scores = []
        new_features = []
        new_masks = []
        for idx in unmatched_box_idx:
            new_boxes.append(boxes[idx])
            new_scores.append(scores[idx])
            new_features.append(pred_features[idx])
            new_masks.append(masks[idx])
        self.add(
            new_boxes, new_scores, new_features, new_masks, distractor_mask
        )


def detect_misjudged_boxes(boxes, next_boxes):
    from torchvision.ops.boxes import _box_inter_union

    inter, union = _box_inter_union(boxes, boxes)
    next_inter, next_union = _box_inter_union(next_boxes, next_boxes)
    box_misjudge_pairs = torch.logical_and(
        inter > 0, torch.logical_and(next_union < union, next_inter < inter)
    )
    misjudged_obj_idxs = []
    for obj_odx, obj_row in enumerate(box_misjudge_pairs):
        neighbours = torch.where(obj_row)[0]
        y_max_obj = boxes[obj_odx, 3]
        y_max_other = boxes[neighbours, 3]
        obj_is_occluded = (y_max_obj < y_max_other).any()
        if obj_is_occluded:
            misjudged_obj_idxs.append(obj_odx)
    return torch.tensor(misjudged_obj_idxs)

