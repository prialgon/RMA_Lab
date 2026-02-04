from enum import Enum
import numpy as np
import numpy.typing

class States(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TURNAROUND = 4

class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    # Robot limits
    LINEAR_SPEED_MAX = 0.22  # Maximum linear velocity in the abscence of angular velocity [m/s]
    SENSOR_RANGE_MIN = 0.16  # Minimum LiDAR sensor range [m]
    SENSOR_RANGE_MAX = 8.0  # Maximum LiDAR sensor range [m]
    TRACK = 0.16  # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.033  # Radius of the wheels [m]
    WHEEL_SPEED_MAX = LINEAR_SPEED_MAX / WHEEL_RADIUS  # Maximum motor angular speed [rad/s]
    
    # LiDAR angles
    INCREMENT = 360/240 * np.pi / 180 # Angle increment [rad]
    LIDAR_X_ANGLE = 15 * np.pi / 180 # Angle for x-axis [rad]
    LIDAR_Z_ANGLE = 45 * np.pi / 180 # Angle for z-axis [rad]
    LIDAR_FILL_ANGLE = 220 * np.pi / 180
    LIDAR_DEADZONE = 10 * np.pi / 180
    ALLOWED_DISTANCE_FORWARD = 0.40 # Distance to objects forward [m]
    ALLOWED_DISTANCE_TURN = 0.30 # Distance to objects on sides [m]
    OBJECTIVE_DISTANCE_WALL = 0.20 # Distance to wall [m]
    
    # PD Control
    SIDE_POINTS = 5 # Number of points to get
    KP = 1.5
    KD = 0.8

    
    def __init__(self, dt: float, logger=None, simulation: bool = False) -> None:
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].
            logger: Logger object to output messages with different severity levels.
            simulation: True if running in simulation, False if running on the real robot.

        """
        self._dt: float = dt
        self._logger = logger
        self._simulation: bool = simulation
        self._current_state = States.FORWARD
        
        self._forward_allowed: bool = False
        self._right_allowed: bool = False
        self._left_allowed: bool = False
        
        self._prev_error: float = 0.0
        
    def _get_lidar_areas(self, ranges: list[float]) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray]:
        # Forward 1st range, 0:x_angle
        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + int(
            self.LIDAR_X_ANGLE / self.INCREMENT
        )
        
        deadzone_start: int = forward_range_1_end
        deadzone_end: int = deadzone_start + int(self.LIDAR_DEADZONE / self.INCREMENT)

        # Left points, x_angle:z_angle
        left_range_start: int = deadzone_end
        left_range_end: int = left_range_start + int(self.LIDAR_Z_ANGLE / self.INCREMENT)

        # Backward Fill points, z_angle:z_angle+2*x_angle
        fill_range_start: int = left_range_end
        fill_range_end: int = fill_range_start + int(
            self.LIDAR_FILL_ANGLE / self.INCREMENT
        )

        # Right points, z_angle+2*x_angle:2*z_angle+2*x_angle
        right_range_start: int = fill_range_end
        right_range_end: int = right_range_start + int(
            self.LIDAR_Z_ANGLE / self.INCREMENT
        )
        
        deadzone_start_2: int = right_range_end
        deadzone_end_2: int = deadzone_start_2 + int(
            self.LIDAR_DEADZONE / self.INCREMENT
        )

        # Forward 2nd range: 2*z_angle+2*x_angle:0
        forward_range_2_start: int = deadzone_end_2

        # Handle NaN
        ranges_array: numpy.typing.NDArray = np.array(ranges)
        np.nan_to_num(ranges_array, copy=False, nan=np.inf)

        forward: numpy.typing.NDArray = np.concatenate(
            (
                ranges_array[forward_range_1_start:forward_range_1_end],
                ranges_array[forward_range_2_start:],
            )
        )

        left: numpy.typing.NDArray = ranges_array[
            left_range_start:left_range_end
        ]
        
        right: numpy.typing.NDArray = ranges_array[
            right_range_start:right_range_end
        ]

        return forward, left, right
    
    def _get_new_state(self, left: float, right: float) -> States:
        if self._current_state in [States.TURNAROUND, States.RIGHT, States.LEFT]:
            if self._forward_allowed:
                return States.FORWARD
            return self._current_state

        if not self._forward_allowed:
            if self._left_allowed and self._right_allowed:
                if left > right:
                    return States.LEFT
                else:
                    return States.RIGHT
            elif self._left_allowed:
                return States.LEFT
            elif self._right_allowed:
                return States.RIGHT
            else:
                return States.TURNAROUND
        
        return States.FORWARD
    
    def _error(self, points: numpy.typing.NDArray) -> float:
        return self.OBJECTIVE_DISTANCE_WALL - float(np.median(points))
    
    def _get_pd(self, error: float) -> float:
        de: float = (error - self._prev_error) / self._dt
        self._prev_error = error
        return self.KP * error + self.KD * de
        
        
    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_scan: Distance from every LiDAR ray to the closest obstacle [m].
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].

        Returns:
            The linear and angular velocity commands for the robot. They must
                v: Linear velocity [m/s].
                w: Angular velocity [rad/s].

        """
        # TODO: 2.14. Complete the function body with your code (i.e., compute v and w).
        v: float = 0.0
        w: float = 0.0
        
        forward, left, right = self._get_lidar_areas(z_scan)
        
        self._forward_allowed = min(forward) > self.ALLOWED_DISTANCE_FORWARD
        self._left_allowed = min(left) > self.ALLOWED_DISTANCE_TURN
        self._right_allowed = min(right) > self.ALLOWED_DISTANCE_TURN
        
        self._current_state = self._get_new_state(min(left), min(right))
        
        if self._current_state == States.FORWARD:
            v = self.LINEAR_SPEED_MAX * 0.8
            right_error: float = self._error(right[:self.SIDE_POINTS])
            w = self._get_pd(right_error)
        elif self._current_state == States.LEFT:
            v = self.LINEAR_SPEED_MAX * 0.6
            w = 1.0
        elif self._current_state == States.RIGHT:
            v = self.LINEAR_SPEED_MAX * 0.6
            w = -1.0
        elif self._current_state == States.TURNAROUND:
            v = 0.0
            w = 1.2
            self._prev_error = 0.0
            
        w_max: float = (self.LINEAR_SPEED_MAX - abs(v)) / (self.TRACK / 2)
        w = np.clip(w, -w_max, w_max)
        
        v_best: float = self.LINEAR_SPEED_MAX - (abs(w) * (self.TRACK / 2))
        
        v = min(v, v_best)
        
        return v, w
    