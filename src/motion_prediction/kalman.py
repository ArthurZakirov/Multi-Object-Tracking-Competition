import cv2
import numpy as np
import torch


class KalmanFilter:
    """

    Parameters
    -----------------------------------------------------------------------------------------------------------
    process_variance 
        - this is the expected variance of the true pedestrian position, caused by movement
        - the higher the pedestrian velocity, the higher this parameter must be

    measurement_variance
        - this is the expected variance of the detection of a NOT moving pedestrian, caused by detector fluctuation
        - the more noizy the trajectory on a micro level is, the higher this parameter must be
    -----------------------------------------------------------------------------------------------------------


    Tuning
    -----------------------------------------------------------------------------------------------------------
    The following section describes how to tune the Kalman filter based on visualization of trajectories. 
    This is not exact science, but it gives decent results pretty quickly.

    Problem 1:
        - Filtered trajectory oscilates very strongly
        -> solution: increase measurement_variance

    Problem 2:
        - Filtered trajectory is too slow and does not catch up to the measured trajectory 
        -> solution: increase process_variance and / or reduce measurement_variance

    Problem 3:
        - Filtered trajectory is almost overlapping with measured trajectory. 
        -> solution: reduce process_variance
    -----------------------------------------------------------------------------------------------------------
    

    
    State
    -----------------------------------------------------------------------------------------------------------
    This Filter smooths and predicts the position and box size together.
    
    The state vector is [cx, cy, w, h, *cx, *cy, *w, *h]
    
    """

    def __init__(self, process_variance=50, measurement_variance=1, dt=1 / 30):
        kalman = cv2.KalmanFilter(8, 4)

        # H
        H = np.zeros((4, 8), dtype=np.float32)
        H[:, :4] = np.diag(np.ones((4), dtype=np.float32))
        kalman.measurementMatrix = H

        # A
        A = np.eye(8, dtype=np.float32)
        A[:4, 4:] = np.eye(4, dtype=np.float32) * dt
        kalman.transitionMatrix = A

        # Q
        Q = np.zeros((8, 8), dtype=np.float32)
        q = process_variance * np.array([(dt ** 4) / 4, (dt ** 3) / 2])
        q_derivative = process_variance * np.array([(dt ** 3) / 2, dt ** 2])
        Q[0, [0, 4]] = q
        Q[1, [1, 5]] = q
        Q[2, [2, 6]] = q
        Q[3, [3, 7]] = q

        Q[4, [0, 4]] = q_derivative
        Q[5, [1, 5]] = q_derivative
        Q[6, [2, 6]] = q_derivative
        Q[7, [3, 7]] = q_derivative
        kalman.processNoiseCov = Q

        # R
        R = measurement_variance * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = R

        self.kalman = kalman

    def reset_state(self):
        self.kalman.errorCovPost = np.zeros((8, 8), dtype=np.float32)
        self.kalman.errorCovPre = np.zeros((8, 8), dtype=np.float32)
        self.kalman.statePre = np.zeros((8, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((8, 1), dtype=np.float32)

    def predict(self, trajectory, future_len=1):
        if isinstance(trajectory, torch.Tensor):
            return_torch = True
            trajectory = trajectory.numpy()
        else:
            return_torch = False

        self.smooth(trajectory)
        pred = []
        initial_pos = trajectory[0]
        for _ in range(future_len):
            next_pos = self.kalman.predict()[:4].squeeze() + initial_pos
            pred.append(next_pos)

        pred = np.stack(pred, axis=0)
        if return_torch:
            pred = torch.from_numpy(pred)
        return pred

    def smooth(self, trajectory):
        """
        Arguments
        ---------
        trajectory : [L, 4]


        Returns
        -------
        trajectory : [L, 4]
        """
        if isinstance(trajectory, torch.Tensor):
            return_torch = True
            trajectory = trajectory.numpy()
        else:
            return_torch = False

        smoothed_trajectory = []
        for position in trajectory:
            predicted_position = (
                self.kalman.predict()[:4].squeeze() + trajectory[0]
            )
            self.kalman.correct(position - trajectory[0])
            smoothed_trajectory.append(predicted_position)
        # smoothed_trajectory.pop(-1)
        # smoothed_trajectory.append(trajectory[-1])

        smoothed_trajectory = np.stack(smoothed_trajectory, axis=0)
        if return_torch:
            smoothed_trajectory = torch.from_numpy(smoothed_trajectory)
        return smoothed_trajectory


def obj_is_moving(trajectory, pixel_per_second_tresh=5, time_treshold=0.5):
    """
    The kalman filter does a good job at smoothing moving trajectories,
    but it has no functionality of classifying a trajectory as completely standing.
    Instead it tries to smooth the micro movements on a micro level.
    
    -> this functionen freezes the trajectory to a single point, if the object does not move
    """
    dt = 1 / 25
    if len(trajectory) > (time_treshold / dt):
        start = trajectory[0, :2]
        end = trajectory[-1, :2]
        pixel_distance = np.linalg.norm(start - end)
        pixel_velocity = pixel_distance / (len(trajectory) * dt)
        return pixel_velocity > pixel_per_second_tresh
    else:
        return True


def freeze_at_mean(traj):
    if len(traj) > 1:
        if isinstance(traj, torch.Tensor):
            return traj.mean(0).repeat(len(traj), 1)
        if isinstance(traj, np.ndarray):
            return np.tile(traj, (len(traj), 1))
    else:
        return traj


class FullFilter(KalmanFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def smooth(self, traj):
        if obj_is_moving(traj):
            return super().smooth(traj)
        else:
            return freeze_at_mean(traj)

