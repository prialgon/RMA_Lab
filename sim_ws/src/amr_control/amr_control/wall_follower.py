from enum import Enum
import numpy as np
import numpy.typing
from rclpy.impl.rcutils_logger import RcutilsLogger


class States(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    FORWARD_LEFT = 4
    FORWARD_RIGHT = 5
    TURNAROUND = 6


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    # Robot limits
    LINEAR_SPEED_MAX = 0.22  # Maximum linear velocity in the abscence of angular velocity [m/s]
    SENSOR_RANGE_MIN = 0.16  # Minimum LiDAR sensor range [m]
    SENSOR_RANGE_MAX = 8.0  # Maximum LiDAR sensor range [m]
    TRACK = 0.16  # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.033  # Radius of the wheels [m]
    WHEEL_SPEED_MAX = LINEAR_SPEED_MAX / WHEEL_RADIUS  # Maximum motor angular speed [rad/s]
    MINIMUM_LINEAR_VELOCITY = 0.5 * LINEAR_SPEED_MAX  # Minimum linear velocity [m/s]

    # LiDAR angles
    LIDAR_FORWARD_ANGLE = 5  # Angle for forward area [º]
    LIDAR_DEADZONE_TOP = 30  # Angle for area between forward and forward diagonal [º]
    LIDAR_DIAGONAL_ANGLE = 5  # Angle for forward diagonal area [º]
    LIDAR_DEADZONE_BOTTOM = 40  # Angle for area between forward diagonal and side
    LIDAR_SIDE_ANGLE = 10  # Angle for side area [º]
    LIDAR_FILL_ANGLE = 180  # Angle for no-read zone [º]

    # Distance control
    MAX_DISTANCE_FORWARD = 0.40  # Distance to objects forward [m]
    MAX_DISTANCE_FORWARD_WHILE_IN_TURN = 0.35  # Distance to objects forward while turning [m]
    MAX_DISTANCE_TURN = 0.30  # Distance to objects on sides [m]
    MAX_DISTANCE_FORWARD_DIAGONAL_TURNING = 0.22  # Distance to wall for diagonal turning [m]
    MAX_DISTANCE_FORWARD_TURNAROUND = 0.18  # Distance to activate forward turnaround [m]
    # FORWARD_TURN_EMERGENCY_DISTANCE = 0.16 # Distance to forward corner [m]

    # PD Control
    PD_OBJECTIVE_DISTANCE_TO_WALL = 0.18  # Distance to wall [m]
    KP = 2.0
    KD = 2.0

    # Velocity Control
    FORWARD_LINEAR_VELOCITY = 0.8 * LINEAR_SPEED_MAX  # Forward linear velocity [m/s]
    FORWARD_LINEAR_TURN_VELOCITY = 0.6 * LINEAR_SPEED_MAX  # Forward linear velocity in turn [m/s]
    TURN_LINEAR_VELOCITY = 0.5 * LINEAR_SPEED_MAX  # Turning linear velocity [m/s]

    # Angular Control
    FORWARD_TURNING_ANGULAR_VELOCITY = 1.0  # Forward turning angular velocity [rad/s]
    TURN_ANGULAR_VELOCITY = 1.4  # Turning angular velocity [rad/s]
    TURNAROUND_ANGULAR_VELOCITY = 2.0  # Turnaroud angular velocity [rad/s]

    # ----- DO NOT CHANGE -----
    # ASSERTIONS FOR CHECKS
    assert 360 == (
        2 * LIDAR_FORWARD_ANGLE
        + 2 * LIDAR_SIDE_ANGLE
        + 2 * LIDAR_DEADZONE_TOP
        + 2 * LIDAR_DEADZONE_BOTTOM
        + 2 * LIDAR_DIAGONAL_ANGLE
        + LIDAR_FILL_ANGLE
    )

    # LiDAR angles
    LIDAR_FORWARD_ANGLE *= np.pi / 180
    LIDAR_DEADZONE_TOP *= np.pi / 180
    LIDAR_DIAGONAL_ANGLE *= np.pi / 180
    LIDAR_DEADZONE_BOTTOM *= np.pi / 180
    LIDAR_SIDE_ANGLE *= np.pi / 180
    LIDAR_FILL_ANGLE *= np.pi / 180
    # --------------------------

    def __init__(
        self, dt: float, logger: RcutilsLogger | None = None, simulation: bool = False
    ) -> None:
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
        self._forward_left_allowed: bool = False
        self._forward_right_allowed: bool = False

        self._right_emergency_turn: bool = False
        self._left_emergency_turn: bool = False

        self._prev_error: float = 0.0
        self._integral_error: float = 0.0

        self.increment: float = 360 / 240 * np.pi / 180

    def _get_lidar_areas(
        self, ranges: list[float]
    ) -> tuple[
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
    ]:
        self.increment = 360 / len(ranges) * np.pi / 180

        # 1. ========== FORWARD FIRST RANGE - 0:x_angle ==========
        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + int(
            self.LIDAR_FORWARD_ANGLE / self.increment
        )
        # ====================================================================

        # 2. ========== LEFT TOP DEADZONE - x_angle:x_angle+deadzone ==========
        deadzone_left_1_start: int = forward_range_1_end
        deadzone_left_1_end: int = deadzone_left_1_start + int(
            self.LIDAR_DEADZONE_TOP / self.increment
        )
        # ====================================================================

        # 3. ========== FORWARD LEFT - x_angle+deadzone:x_left_forward ==========
        forward_left_range_start: int = deadzone_left_1_end
        forward_left_range_end: int = forward_left_range_start + int(
            self.LIDAR_DIAGONAL_ANGLE / self.increment
        )
        # ====================================================================

        # 4. ========== LEFT DOWN DEADZONE - x_left_forward:x_left_forward+deadzone ==========
        deadzone_left_2_start: int = forward_left_range_end
        deadzone_left_2_end: int = deadzone_left_2_start + int(
            self.LIDAR_DEADZONE_BOTTOM / self.increment
        )
        # ====================================================================

        # 5. ========== LEFT - x_left_forward+deadzone:x_left ==========
        left_range_start: int = deadzone_left_2_end
        left_range_end: int = left_range_start + int(self.LIDAR_SIDE_ANGLE / self.increment)
        # ====================================================================

        # 6. ========== FILL - 90:270 (Back of robot) ==========
        fill_range_start: int = left_range_end
        fill_range_end: int = fill_range_start + int(self.LIDAR_FILL_ANGLE / self.increment)
        # ====================================================================

        # 7. ========== RIGHT - fill:x_right ==========
        right_range_start: int = fill_range_end
        right_range_end: int = right_range_start + int(self.LIDAR_SIDE_ANGLE / self.increment)
        # ====================================================================

        # 8. ========== RIGHT DOWN DEADZONE - x_right:x_right+deadzone ==========
        deadzone_right_2_start: int = right_range_end
        deadzone_right_2_end: int = deadzone_right_2_start + int(
            self.LIDAR_DEADZONE_BOTTOM / self.increment
        )
        # ====================================================================

        # 9. ========== FORWARD RIGHT - x_right+deadzone:x_right_forward ==========
        forward_right_range_start: int = deadzone_right_2_end
        forward_right_range_end: int = forward_right_range_start + int(
            self.LIDAR_DIAGONAL_ANGLE / self.increment
        )
        # ====================================================================

        # 10. ========== RIGHT TOP DEADZONE - x_right_forward:x_right_forward+deadzone ==========
        deadzone_right_1_start: int = forward_right_range_end
        deadzone_right_1_end: int = deadzone_right_1_start + int(
            self.LIDAR_DEADZONE_TOP / self.increment
        )
        # ====================================================================

        # 11. ========== FORWARD SECOND RANGE - x_right_forward+deadzone:360 ==========
        forward_range_2_start: int = deadzone_right_1_end
        # ====================================================================

        # Handle NaN and Infinities
        ranges_array: numpy.typing.NDArray = np.array(ranges)
        ranges_array = np.nan_to_num(
            ranges_array,
            nan=self.SENSOR_RANGE_MIN,
            posinf=self.SENSOR_RANGE_MAX,
            neginf=self.SENSOR_RANGE_MIN,
        )
        ranges_array = np.clip(
            ranges_array, a_min=self.SENSOR_RANGE_MIN, a_max=self.SENSOR_RANGE_MAX
        )

        # Create slices
        forward: numpy.typing.NDArray = np.concatenate(
            (
                ranges_array[forward_range_1_start:forward_range_1_end],
                ranges_array[forward_range_2_start:],
            )
        )

        left: numpy.typing.NDArray = ranges_array[left_range_start:left_range_end]

        right: numpy.typing.NDArray = ranges_array[right_range_start:right_range_end]

        forward_left: numpy.typing.NDArray = ranges_array[
            forward_left_range_start:forward_left_range_end
        ]

        forward_right: numpy.typing.NDArray = ranges_array[
            forward_right_range_start:forward_right_range_end
        ]

        return forward, left, right, forward_left, forward_right

    def _get_new_state(
        self, left: float, right: float, forward_left: float, forward_right: float
    ) -> States:
        if not self._left_allowed and not self._right_allowed and not self._forward_allowed:
            return States.TURNAROUND

        if self._current_state in [States.TURNAROUND, States.RIGHT, States.LEFT]:
            if self._forward_allowed:
                return States.FORWARD
            return self._current_state

        # if self._left_emergency_turn and self._right_emergency_turn:
        #     if left > right:
        #         return States.LEFT
        #     else:
        #         return States.RIGHT
        # elif self._left_emergency_turn:
        #     return States.LEFT
        # elif self._right_emergency_turn:
        #     return States.RIGHT

        if self._forward_allowed:
            return States.FORWARD

        if self._forward_left_allowed and self._forward_right_allowed:
            if forward_left > forward_right:
                return States.FORWARD_LEFT
            else:
                return States.FORWARD_RIGHT
        elif self._forward_left_allowed:
            return States.FORWARD_LEFT
        elif self._forward_right_allowed:
            return States.FORWARD_RIGHT

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

        return self._current_state

    def _error(self, points: np.ndarray) -> float:
        return float(np.median(points)) - self.PD_OBJECTIVE_DISTANCE_TO_WALL

    def _get_pd(self, points: np.ndarray) -> float:
        error: float = self._error(points)
        de = (error - self._prev_error) / self._dt
        self._prev_error = error
        return -(self.KP * error + self.KD * de)

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

        # Logging
        # if self._logger is not None:
        #     self._logger.warn(f"STATE - {self._current_state}")

        v: float = 0.0
        w: float = 0.0

        forward, left, right, forward_left, forward_right = self._get_lidar_areas(z_scan)

        if (
            len(forward) < 1
            or len(left) < 1
            or len(right) < 1
            or len(forward_left) < 1
            or len(forward_right) < 1
        ):
            return v, w

        # Forward allowed
        if self._current_state not in [States.LEFT, States.RIGHT]:
            self._forward_allowed = bool(np.median(forward) > self.MAX_DISTANCE_FORWARD)
            self._forward_turning = False
        else:  # Forward turning allowed
            self._forward_allowed = bool(
                np.median(forward) > self.MAX_DISTANCE_FORWARD_WHILE_IN_TURN
            )
            self._forward_turning = True

        # Forward turning allowed
        self._forward_left_allowed = bool(
            np.median(forward_left) > self.MAX_DISTANCE_FORWARD_DIAGONAL_TURNING
        )
        self._forward_right_allowed = bool(
            np.median(forward_right) > self.MAX_DISTANCE_FORWARD_DIAGONAL_TURNING
        )

        # Turning allowed
        self._left_allowed = bool(np.median(left) > self.MAX_DISTANCE_TURN)
        self._right_allowed = bool(np.median(right) > self.MAX_DISTANCE_TURN)

        # Emergency turn
        # self._left_emergency_turn = np.median(forward_right) < self.FORWARD_TURN_EMERGENCY_DISTANCE
        # self._right_emergency_turn = np.median(forward_left) < self.FORWARD_TURN_EMERGENCY_DISTANCE

        self._current_state = self._get_new_state(
            max(left), max(right), float(np.median(forward_left)), float(np.median(forward_right))
        )

        # If danger forward and not turning --> Turnaround
        if min(forward) < self.MAX_DISTANCE_FORWARD_TURNAROUND and self._current_state not in [
            States.LEFT,
            States.RIGHT,
        ]:
            self._current_state = States.TURNAROUND

        # ========== STATE MACHINE ==========
        match self._current_state:
            # ----- FORWARD -----
            case States.FORWARD:
                if not self._forward_turning:
                    v = self.FORWARD_LINEAR_VELOCITY
                else:
                    v = self.FORWARD_LINEAR_TURN_VELOCITY

                w = self._get_pd(right)

            # ----- FORWARD LEFT -----
            case States.FORWARD_LEFT:
                v = self.FORWARD_LINEAR_TURN_VELOCITY
                w = self.FORWARD_TURNING_ANGULAR_VELOCITY

            # ----- FORWARD RIGHT -----
            case States.FORWARD_RIGHT:
                v = self.FORWARD_LINEAR_TURN_VELOCITY
                w = -self.FORWARD_TURNING_ANGULAR_VELOCITY

            # ----- LEFT -----
            case States.LEFT:
                v = self.TURN_LINEAR_VELOCITY
                w = self.TURN_ANGULAR_VELOCITY

            # ----- RIGHT -----
            case States.RIGHT:
                v = self.TURN_LINEAR_VELOCITY
                w = -self.TURN_ANGULAR_VELOCITY

            # ----- TURNAROUND -----
            case States.TURNAROUND:
                v = 0.0
                w = self.TURNAROUND_ANGULAR_VELOCITY

        # ---- LIMIT W and V -----
        w_max: float = (self.LINEAR_SPEED_MAX - abs(min(self.MINIMUM_LINEAR_VELOCITY, v))) / (
            self.TRACK / 2
        )
        w = np.clip(w, -w_max, w_max)

        v_best: float = self.LINEAR_SPEED_MAX - (abs(w) * (self.TRACK / 2))
        v = min(v, v_best)

        return v, w
