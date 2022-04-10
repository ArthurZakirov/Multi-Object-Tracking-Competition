import cv2
import numpy as np
import torch

# TODO smooth box size aswell


class ConstVelocityFilter:
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
    
    """

    def __init__(self, process_variance=50, measurement_variance=1, dt=1 / 30):
        kalman = cv2.KalmanFilter(4, 2)

        # H
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )

        # A
        kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )

        # Q
        kalman.processNoiseCov = process_variance * np.array(
            [
                [(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                [(dt ** 3) / 2, 0, dt ** 2, 0],
                [0, (dt ** 3) / 2, 0, dt ** 2],
            ],
            np.float32,
        )

        # R
        kalman.measurementNoiseCov = measurement_variance * np.array(
            [[1, 0], [0, 1]], np.float32
        )

        self.kalman = kalman

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
            next_pos = self.kalman.predict()[:2].squeeze() + initial_pos
            pred.append(next_pos)

        pred = np.stack(pred, axis=0)
        if return_torch:
            pred = torch.from_numpy(pred)
        return pred

    def smooth(self, trajectory):
        """
        Arguments
        ---------
        trajectory : [L, 2]


        Returns
        -------
        trajectory : [L, 2]
        """
        if isinstance(trajectory, torch.Tensor):
            return_torch = True
            trajectory = trajectory.numpy()
        else:
            return_torch = False

        smoothed_trajectory = []
        for position in trajectory:
            predicted_position = (
                self.kalman.predict()[:2].squeeze() + trajectory[0]
            )
            self.kalman.correct(position - trajectory[0])
            smoothed_trajectory.append(predicted_position)

        smoothed_trajectory = np.stack(smoothed_trajectory, axis=0)
        if return_torch:
            smoothed_trajectory = torch.from_numpy(smoothed_trajectory)
        return smoothed_trajectory


class FullBoxFilter:
    """

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


class ConstAccelerationFilter:
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
        

    
    """

    def __init__(self, process_variance=50, measurement_variance=1, dt=1 / 30):
        kalman = cv2.KalmanFilter(6, 2)

        # H
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], np.float32
        )

        # A
        kalman.transitionMatrix = np.array(
            [
                [1, 0, dt, 0, (dt ** 2) / 2, 0],
                [0, 1, 0, dt, 0, (dt ** 2) / 2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        # Q
        kalman.processNoiseCov = process_variance * np.array(
            [
                [(dt ** 4) / 4, 0, (dt ** 3) / 2, 0, (dt ** 2) / 2, 0],
                [0, (dt ** 4) / 4, 0, (dt ** 3) / 2, 0, (dt ** 2) / 2],
                [(dt ** 3) / 2, 0, dt ** 2, 0, dt, 0],
                [0, (dt ** 3) / 2, 0, dt ** 2, 0, dt],
                [(dt ** 2) / 2, 0, dt, 0, 1, 0],
                [0, (dt ** 2) / 2, 0, dt, 0, 1],
            ],
            np.float32,
        )

        # R
        kalman.measurementNoiseCov = measurement_variance * np.array(
            [[1, 0], [0, 1]], np.float32
        )

        self.kalman = kalman

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
            next_pos = self.kalman.predict()[:2].squeeze() + initial_pos
            pred.append(next_pos)

        pred = np.stack(pred, axis=0)
        if return_torch:
            pred = torch.from_numpy(pred)
        return pred

    def smooth(self, trajectory):
        """
        Arguments
        ---------
        trajectory : [L, 2]


        Returns
        -------
        trajectory : [L, 2]
        """
        if isinstance(trajectory, torch.Tensor):
            return_torch = True
            trajectory = trajectory.numpy()
        else:
            return_torch = False

        smoothed_trajectory = []
        for position in trajectory:
            predicted_position = (
                self.kalman.predict()[:2].squeeze() + trajectory[0]
            )
            self.kalman.correct(position - trajectory[0])
            smoothed_trajectory.append(predicted_position)
        # smoothed_trajectory.pop(-1)
        # smoothed_trajectory.append(trajectory[-1])

        smoothed_trajectory = np.stack(smoothed_trajectory, axis=0)
        if return_torch:
            smoothed_trajectory = torch.from_numpy(smoothed_trajectory)
        return smoothed_trajectory

