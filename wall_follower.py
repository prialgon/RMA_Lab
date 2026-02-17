from enum import Enum
import numpy as np
import numpy.typing
from rclpy.impl.rcutils_logger import RcutilsLogger

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
    LIDAR_X_ANGLE = 10 * np.pi / 180 
    LIDAR_Z_ANGLE = 10 * np.pi / 180 
    LIDAR_FILL_ANGLE = 180 * np.pi / 180 
    
    # New Corner Slicing
    LIDAR_DEADZONE = 70 * np.pi / 180 
    LIDAR_CORNER_ANGLE = 5 * np.pi / 180 # The 5-degree measurement zone
    # Calculate the remaining deadzone to split it evenly around the corner slice
    LIDAR_DEADZONE_SLICE = (LIDAR_DEADZONE - LIDAR_CORNER_ANGLE) / 2
    
    # Distance control
    OBJECTIVE_DISTANCE_WALL = 0.22 # Distance to wall [m]
    ALLOWED_DISTANCE_FORWARD = 0.40 # Distance to objects forward [m]
    ALLOWED_DISTANCE_FORWARD_IN_TURN = 0.30 # Distance to objects forward [m]
    ALLOWED_DISTANCE_TURN = 0.25 # Distance to objects on sides [m]
    EMERGENCY_DISTANCE_TURNAROUND = 0.18 # Distance to activate turnaround [m]
    FORWARD_TURN_EMERGENCY_DISTANCE = 0.16 # Distance to forward corner [m]
    FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND = 0.20 # Distance to all walls [m]
    
    # PD Control
    SIDE_POINTS = 10 # Number of points to get
    KP = 2.0
    KD = 0.2
    KI = 0.3

    # Velocity Control
    MINIMUM_LINEAR_VELOCITY = 0.5
    FORWARD_LINEAR_VELOCITY = 0.99 * LINEAR_SPEED_MAX # Forward linear velocity [m/s]
    FORWARD_LINEAR_TURN_VELOCITY = 0.8 * LINEAR_SPEED_MAX # Forward linear velocity while turning [m/s]
    TURN_LINEAR_VELOCITY = 0.8 * LINEAR_SPEED_MAX # Turning linear velocity [m/s]

    # Angular Control
    TURN_ANGULAR_VELOCITY = 4.0 # Turning angular velocity [rad/s]
    TURNAROUND_ANGULAR_VELOCITY = 5.5 # Turnaroud angular velocity [rad/s]

    
    def __init__(self, dt: float, logger: RcutilsLogger | None = None, simulation: bool = False) -> None:
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].
            logger: Logger object to output messages with different severity levels.
            simulation: True if running in simulation, False if running on the real robot.

        """
        self._dt: float = dt
        self._logger: RcutilsLogger | None = logger
        self._simulation: bool = simulation 
        self._current_state = States.FORWARD
        
        self._forward_allowed: bool = False
        self._right_allowed: bool = False
        self._left_allowed: bool = False
        self._forward_turning: bool = False
        
        self._right_emergency_turn: bool = False
        self._left_emergency_turn: bool = False
        
        self._prev_error: float = 0.0
        self._integral_error: float = 0.0
        
        self.increment: float = 360/240 * np.pi / 180
        
    def _get_lidar_areas(self, ranges: list[float]) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray]:
        # Forward 1st range, 0:x_angle
        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + int(
            self.LIDAR_X_ANGLE / self.increment
        )
        
        forward_left_range_start: int = forward_range_1_end
        forward_left_range_end: int = forward_left_range_start + int(self.LIDAR_DEADZONE / self.increment)

        # Left points, x_angle:z_angle
        left_range_start: int = forward_left_range_end
        left_range_end: int = left_range_start + int(self.LIDAR_Z_ANGLE / self.increment)

        # Backward Fill points, z_angle:z_angle+2*x_angle
        fill_range_start: int = left_range_end
        fill_range_end: int = fill_range_start + int(
            self.LIDAR_FILL_ANGLE / self.increment
        )

        # Right points, z_angle+2*x_angle:2*z_angle+2*x_angle
        right_range_start: int = fill_range_end
        right_range_end: int = right_range_start + int(
            self.LIDAR_Z_ANGLE / self.increment
        )
        
        forward_right_range_start: int = right_range_end
        forward_right_range_end: int = forward_right_range_start + int(
            self.LIDAR_DEADZONE / self.increment
        )

        # Forward 2nd range: 2*z_angle+2*x_angle:0
        forward_range_2_start: int = forward_right_range_end

        # Handle NaN
        ranges_array: numpy.typing.NDArray = np.array(ranges)
        np.nan_to_num(ranges_array, copy=False, nan=self.SENSOR_RANGE_MIN)

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
        
        forward_left: numpy.typing.NDArray = ranges_array[
            forward_left_range_start:forward_left_range_end
        ]
        
        forward_right: numpy.typing.NDArray = ranges_array[
            forward_right_range_start:forward_right_range_end
        ]

        return forward, left, right, forward_left, forward_right
    
    def _get_new_state(self, left: float, right: float) -> States:
        if self._current_state in [States.TURNAROUND, States.RIGHT, States.LEFT]:
            if self._forward_allowed:
                return States.FORWARD
            return self._current_state
        
        if self._left_emergency_turn and self._right_emergency_turn:
            if left > right:
                return States.LEFT
            else:
                return States.RIGHT
        elif self._left_emergency_turn:
            return States.LEFT
        elif self._right_emergency_turn:
            return States.RIGHT
            
        
        if self._forward_allowed:
            return States.FORWARD
        
        if self._current_state == States.FORWARD:
            if self._right_allowed and self._left_allowed:
                if left > right:
                    return States.LEFT
                else:
                    return States.RIGHT
            
            if self._right_allowed:
                return States.RIGHT
            
            if self._left_allowed:
                return States.LEFT

            if not self._left_allowed and not self._right_allowed:
                return States.TURNAROUND
        
        raise Exception("NO NEW STATE")
        
    
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
        self.increment = 360/len(z_scan) * np.pi / 180 # Angle increment [rad]
        
        v: float = 0.0
        w: float = 0.0
        
        forward, left, right, forward_left, forward_right = self._get_lidar_areas(z_scan)
        
        if len(forward) < 1 or len(left) < 1 or len(right) < 1 or len(forward_left) < 1 or len(forward_right) < 1:
            return v, w
        
        # Forward allowed
        if self._current_state not in [States.LEFT, States.RIGHT]:
            self._forward_allowed = min(forward) > self.ALLOWED_DISTANCE_FORWARD
            self._forward_turning = False
        else: # Forward turning allowed
            self._forward_allowed = min(forward) > self.ALLOWED_DISTANCE_FORWARD_IN_TURN
            self._forward_turning = True
        
        # Turning allowed
        self._left_allowed = min(left) > self.ALLOWED_DISTANCE_TURN
        self._right_allowed = min(right) > self.ALLOWED_DISTANCE_TURN
        
        # Turnaround in no exit corridor
        if (all(forward < self.FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND) and 
            all(left < self.FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND) and 
            all(right < self.FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND)):
            self._current_state = States.TURNAROUND
        
        # Emergency turn
        self._left_emergency_turn = min(forward_right) < self.FORWARD_TURN_EMERGENCY_DISTANCE
        self._right_emergency_turn = min(forward_left) < self.FORWARD_TURN_EMERGENCY_DISTANCE
        
        self._current_state = self._get_new_state(max(left), max(right))
        
        if min(forward) < self.EMERGENCY_DISTANCE_TURNAROUND:
            self._current_state = States.TURNAROUND
        
        # Logging
        if self._logger is not None:
            # self._logger.warn(str(np.median(right[:self.SIDE_POINTS])))
            self._logger.warn(str(self._current_state))
        
        if self._current_state == States.FORWARD:
            if not self._forward_turning:
                v = self.FORWARD_LINEAR_VELOCITY
            else:
                v = self.FORWARD_LINEAR_TURN_VELOCITY
                
            right_error: float = self._error(right)
            w = self._get_pd(right_error)
        elif self._current_state == States.LEFT:
            v = self.TURN_LINEAR_VELOCITY
            w = self.TURN_ANGULAR_VELOCITY
        elif self._current_state == States.RIGHT:
            v = self.TURN_LINEAR_VELOCITY
            w = -self.TURN_ANGULAR_VELOCITY
        elif self._current_state == States.TURNAROUND:
            v = 0.0
            w = self.TURNAROUND_ANGULAR_VELOCITY
            
        self._prev_error = self._error(right)
            
        w_max: float = (self.LINEAR_SPEED_MAX - abs(min(self.MINIMUM_LINEAR_VELOCITY, v))) / (self.TRACK / 2)
        w = np.clip(w, -w_max, w_max)
        
        v_best: float = self.LINEAR_SPEED_MAX - (abs(w) * (self.TRACK / 2))
        
        v = min(v, v_best)
        
        return v, w
    
