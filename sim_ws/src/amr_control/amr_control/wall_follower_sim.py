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

    # Robot limits.
    LINEAR_SPEED_MAX = 0.22  # Max linear velocity (no angular velocity) [m/s].
    SENSOR_RANGE_MIN = 0.16  # Min LiDAR range [m].
    SENSOR_RANGE_MAX = 8.0   # Max LiDAR range [m].
    TRACK = 0.16             # Wheel-to-wheel distance [m].
    WHEEL_RADIUS = 0.033     # Wheel radius [m].
    # Max wheel speed [rad/s].
    WHEEL_SPEED_MAX = LINEAR_SPEED_MAX / WHEEL_RADIUS

    # LiDAR angles.
    LIDAR_X_ANGLE = 5 * np.pi / 180       # Forward slice half-angle [rad].
    LIDAR_Z_ANGLE = 5 * np.pi / 180       # Side slice half-angle [rad].
    LIDAR_FILL_ANGLE = 180 * np.pi / 180  # Back (ignored) zone [rad].
    # Gap between forward and side zones [rad].
    LIDAR_DEADZONE = 70 * np.pi / 180

    # Corner slice configuration.
    LIDAR_CORNER_ANGLE = 5 * np.pi / 180
    LIDAR_DEADZONE_SLICE = (LIDAR_DEADZONE - LIDAR_CORNER_ANGLE) / 2

    # Distance thresholds [m].
    OBJECTIVE_DISTANCE_WALL = 0.20
    ALLOWED_DISTANCE_FORWARD = 0.40
    ALLOWED_DISTANCE_FORWARD_IN_TURN = 0.30
    ALLOWED_DISTANCE_TURN = 0.25
    EMERGENCY_DISTANCE_TURNAROUND = 0.18
    FORWARD_TURN_EMERGENCY_DISTANCE = 0.16
    FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND = 0.20

    # PD control.
    SIDE_POINTS = 10
    KP = 1.5
    KD = 0.8

    # Linear velocities [m/s].
    FORWARD_LINEAR_VELOCITY = 0.99 * LINEAR_SPEED_MAX
    FORWARD_LINEAR_TURN_VELOCITY = 0.8 * LINEAR_SPEED_MAX
    TURN_LINEAR_VELOCITY = 0.6 * LINEAR_SPEED_MAX

    # Angular velocities [rad/s].
    TURN_ANGULAR_VELOCITY = 0.8 * WHEEL_SPEED_MAX
    TURNAROUND_ANGULAR_VELOCITY = 0.99 * WHEEL_SPEED_MAX

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
        self.increment: float = 360 / 240 * np.pi / 180

    def _get_lidar_areas(self, ranges: list[float]) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray]:
        self.increment = 360 / len(list) * np.pi / 180

        # Forward slice (first half, ahead of the robot).
        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + \
            int(self.LIDAR_X_ANGLE / self.increment)

        # Left top deadzone.
        deadzone_left_1_start: int = forward_range_1_end
        deadzone_left_1_end: int = deadzone_left_1_start + \
            int(self.LIDAR_DEADZONE / self.increment)

        # Forward-left slice.
        forward_left_range_start: int = deadzone_left_1_end
        forward_left_range_end: int = forward_left_range_start + \
            int(self.LIDAR_XZ_ANGLE / self.increment)

        # Left bottom deadzone.
        deadzone_left_2_start: int = forward_left_range_end
        deadzone_left_2_end: int = deadzone_left_2_start + \
            int(self.LIDAR_DEADZONE / self.increment)

        # Left slice.
        left_range_start: int = deadzone_left_2_end
        left_range_end: int = left_range_start + \
            int(self.LIDAR_Z_ANGLE / self.increment)

        # Rear fill (ignored).
        fill_range_start: int = left_range_end
        fill_range_end: int = fill_range_start + \
            int(self.LIDAR_FILL_ANGLE / self.increment)

        # Right slice.
        right_range_start: int = fill_range_end
        right_range_end: int = right_range_start + \
            int(self.LIDAR_Z_ANGLE / self.increment)

        # Right bottom deadzone.
        deadzone_right_2_start: int = right_range_end
        deadzone_right_2_end: int = deadzone_right_2_start + \
            int(self.LIDAR_DEADZONE / self.increment)

        # Forward-right slice.
        forward_right_range_start: int = deadzone_right_2_end
        forward_right_range_end: int = forward_right_range_start + \
            int(self.LIDAR_XZ_ANGLE / self.increment)

        # Right top deadzone.
        deadzone_right_1_start: int = forward_right_range_end
        deadzone_right_1_end: int = deadzone_right_1_start + \
            int(self.LIDAR_DEADZONE / self.increment)

        # Forward slice (second half, wrapping around).
        forward_range_2_start: int = deadzone_right_1_end

        # Clean NaN and infinities from the ranges.
        ranges_array: numpy.typing.NDArray = np.array(ranges)
        ranges_array = np.nan_to_num(ranges_array, nan=self.SENSOR_RANGE_MIN,
                                     posinf=self.SENSOR_RANGE_MAX, neginf=self.SENSOR_RANGE_MIN)
        ranges_array = np.clip(
            ranges_array, a_min=self.SENSOR_RANGE_MIN, a_max=self.SENSOR_RANGE_MAX)

        # Build each slice.
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
        v: float = 0.0
        w: float = 0.0

        forward, left, right, forward_left, forward_right = self._get_lidar_areas(
            z_scan)

        # Is moving forward allowed at the current state?
        if self._current_state not in [States.LEFT, States.RIGHT]:
            self._forward_allowed = min(
                forward) > self.ALLOWED_DISTANCE_FORWARD
            self._forward_turning = False
        else:
            self._forward_allowed = min(
                forward) > self.ALLOWED_DISTANCE_FORWARD_IN_TURN
            self._forward_turning = True

        # Is turning allowed on each side?
        self._left_allowed = min(left) > self.ALLOWED_DISTANCE_TURN
        self._right_allowed = min(right) > self.ALLOWED_DISTANCE_TURN

        # Dead-end corridor: force a turnaround.
        if (all(forward < self.FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND) and
            all(left < self.FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND) and
                all(right < self.FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND)):
            self._current_state = States.TURNAROUND

        # Emergency turn triggered by a close forward-corner obstacle.
        self._left_emergency_turn = min(
            forward_right) < self.FORWARD_TURN_EMERGENCY_DISTANCE
        self._right_emergency_turn = min(
            forward_left) < self.FORWARD_TURN_EMERGENCY_DISTANCE

        self._current_state = self._get_new_state(max(left), max(right))

        if min(forward) < self.EMERGENCY_DISTANCE_TURNAROUND:
            self._current_state = States.TURNAROUND

        if self._current_state == States.FORWARD:
            if not self._forward_turning:
                v = self.FORWARD_LINEAR_VELOCITY
            else:
                v = self.FORWARD_LINEAR_TURN_VELOCITY

            right_error: float = self._error(right[:self.SIDE_POINTS])
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

        self._prev_error = self._error(right[:self.SIDE_POINTS])

        w_max: float = (self.LINEAR_SPEED_MAX -
                        abs(min(self.MINIMUM_LINEAR_VELOCITY, v))) / (self.TRACK / 2)
        old_w_max = w_max
        w = np.clip(w, -w_max, w_max)

        v_best: float = self.LINEAR_SPEED_MAX - (abs(w) * (self.TRACK / 2))

        if self._logger is not None:
            self._logger.warn(
                f"Linear velocity HIGH - {v > v_best}, Angular velocity HIGH - {abs(old_w_max) > abs(w)}, STATE - {self._current_state}")

        v = min(v, v_best)

        return v, w
