from enum import Enum
import numpy as np
import numpy.typing
from rclpy.impl.rcutils_logger import RcutilsLogger


class States(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TURNAROUND = 4
    FORWARD_LEFT = 5
    FORWARD_RIGHT = 6


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    # Robot limits (common)
    LINEAR_SPEED_MAX = 0.22
    SENSOR_RANGE_MIN = 0.16
    SENSOR_RANGE_MAX = 8.0
    TRACK = 0.16
    WHEEL_RADIUS = 0.033
    WHEEL_SPEED_MAX = LINEAR_SPEED_MAX / WHEEL_RADIUS

    # Simulation configuration
    SIM_INCREMENT = 360 / 240 * np.pi / 180
    SIM_LIDAR_X_ANGLE = 18 * np.pi / 180
    SIM_LIDAR_Z_ANGLE = 70 * np.pi / 180
    SIM_LIDAR_FILL_ANGLE = 180 * np.pi / 180
    SIM_LIDAR_DEADZONE = 2 * np.pi / 180

    SIM_OBJECTIVE_DISTANCE_WALL = 0.20
    SIM_ALLOWED_DISTANCE_FORWARD = 0.50
    SIM_ALLOWED_DISTANCE_FORWARD_IN_TURN = 0.25
    SIM_ALLOWED_DISTANCE_TURN = 0.15
    SIM_EMERGENCY_DISTANCE_TURNAROUND = 0.18
    SIM_FORWARD_TURN_EMERGENCY_DISTANCE = 0.05
    SIM_FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND = 0.45

    SIM_SIDE_POINTS = 10
    SIM_KP = 1.5
    SIM_KD = 0.8

    SIM_FORWARD_LINEAR_VELOCITY = 0.99 * LINEAR_SPEED_MAX
    SIM_FORWARD_LINEAR_TURN_VELOCITY = 0.9 * LINEAR_SPEED_MAX
    SIM_TURN_LINEAR_VELOCITY = 0.6 * LINEAR_SPEED_MAX
    SIM_TURN_ANGULAR_VELOCITY = 0.8 * WHEEL_SPEED_MAX
    SIM_TURNAROUND_ANGULAR_VELOCITY = 0.99 * WHEEL_SPEED_MAX

    # Real robot configuration
    REAL_LIDAR_X_ANGLE = 5 * np.pi / 180
    REAL_LIDAR_Z_ANGLE = 10 * np.pi / 180
    REAL_LIDAR_FILL_ANGLE = 180 * np.pi / 180
    REAL_LIDAR_DEADZONE = 32.5 * np.pi / 180
    REAL_LIDAR_XZ_ANGLE = 10 * np.pi / 180

    REAL_OBJECTIVE_DISTANCE_WALL = 0.17
    REAL_ALLOWED_DISTANCE_FORWARD = 0.35
    REAL_ALLOWED_DISTANCE_FORWARD_IN_TURN = 0.28
    REAL_ALLOWED_DISTANCE_TURN = 0.25
    REAL_ALLOWED_DISTANCE_FORWARD_TURN = 0.22

    REAL_KP = 1.5
    REAL_KD = 2.0

    REAL_MINIMUM_LINEAR_VELOCITY = 0.3 * LINEAR_SPEED_MAX
    REAL_FORWARD_LINEAR_VELOCITY = 0.7 * LINEAR_SPEED_MAX
    REAL_FORWARD_LINEAR_TURN_VELOCITY = 0.4 * LINEAR_SPEED_MAX
    REAL_TURN_LINEAR_VELOCITY = 0.3 * LINEAR_SPEED_MAX
    REAL_FORWARD_TURNING_ANGULAR_VELOCITY = 1.4
    REAL_TURN_ANGULAR_VELOCITY = 1.9
    REAL_TURNAROUND_ANGULAR_VELOCITY = 1.8

    def __init__(
        self, dt: float, logger: RcutilsLogger | None = None, simulation: bool = False
    ) -> None:
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

        self._prev_error: float | None = 0.0

        self.increment: float = self.SIM_INCREMENT

    def _get_lidar_areas(self, ranges: list[float]) -> tuple[
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
    ]:
        if self._simulation:
            return self._get_lidar_areas_sim(ranges)
        return self._get_lidar_areas_real(ranges)

    def _get_lidar_areas_sim(self, ranges: list[float]) -> tuple[
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
    ]:
        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + int(
            self.SIM_LIDAR_X_ANGLE / self.SIM_INCREMENT
        )

        forward_left_range_start: int = forward_range_1_end
        forward_left_range_end: int = forward_left_range_start + int(
            self.SIM_LIDAR_DEADZONE / self.SIM_INCREMENT
        )

        left_range_start: int = forward_left_range_end
        left_range_end: int = left_range_start + int(
            self.SIM_LIDAR_Z_ANGLE / self.SIM_INCREMENT
        )

        fill_range_start: int = left_range_end
        fill_range_end: int = fill_range_start + int(
            self.SIM_LIDAR_FILL_ANGLE / self.SIM_INCREMENT
        )

        right_range_start: int = fill_range_end
        right_range_end: int = right_range_start + int(
            self.SIM_LIDAR_Z_ANGLE / self.SIM_INCREMENT
        )

        forward_right_range_start: int = right_range_end
        forward_right_range_end: int = forward_right_range_start + int(
            self.SIM_LIDAR_DEADZONE / self.SIM_INCREMENT
        )

        forward_range_2_start: int = forward_right_range_end

        ranges_array: numpy.typing.NDArray = np.array(ranges)
        np.nan_to_num(ranges_array, copy=False, nan=np.inf)

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

    def _get_lidar_areas_real(self, ranges: list[float]) -> tuple[
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
        numpy.typing.NDArray,
    ]:
        self.increment = 360 / len(ranges) * np.pi / 180

        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + int(
            self.REAL_LIDAR_X_ANGLE / self.increment
        )

        deadzone_left_1_start: int = forward_range_1_end
        deadzone_left_1_end: int = deadzone_left_1_start + int(
            self.REAL_LIDAR_DEADZONE / self.increment
        )

        forward_left_range_start: int = deadzone_left_1_end
        forward_left_range_end: int = forward_left_range_start + int(
            self.REAL_LIDAR_XZ_ANGLE / self.increment
        )

        deadzone_left_2_start: int = forward_left_range_end
        deadzone_left_2_end: int = deadzone_left_2_start + int(
            self.REAL_LIDAR_DEADZONE / self.increment
        )

        left_range_start: int = deadzone_left_2_end
        left_range_end: int = left_range_start + int(
            self.REAL_LIDAR_Z_ANGLE / self.increment
        )

        fill_range_start: int = left_range_end
        fill_range_end: int = fill_range_start + int(
            self.REAL_LIDAR_FILL_ANGLE / self.increment
        )

        right_range_start: int = fill_range_end
        right_range_end: int = right_range_start + int(
            self.REAL_LIDAR_Z_ANGLE / self.increment
        )

        deadzone_right_2_start: int = right_range_end
        deadzone_right_2_end: int = deadzone_right_2_start + int(
            self.REAL_LIDAR_DEADZONE / self.increment
        )

        forward_right_range_start: int = deadzone_right_2_end
        forward_right_range_end: int = forward_right_range_start + int(
            self.REAL_LIDAR_XZ_ANGLE / self.increment
        )

        deadzone_right_1_start: int = forward_right_range_end
        deadzone_right_1_end: int = deadzone_right_1_start + int(
            self.REAL_LIDAR_DEADZONE / self.increment
        )

        forward_range_2_start: int = deadzone_right_1_end

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
        self,
        left: float,
        right: float,
        forward_left: float | None = None,
        forward_right: float | None = None,
    ) -> States:
        if self._simulation:
            return self._get_new_state_sim(left, right)
        return self._get_new_state_real(left, right, forward_left, forward_right)

    def _get_new_state_sim(self, left: float, right: float) -> States:
        if self._current_state in [States.TURNAROUND, States.RIGHT, States.LEFT]:
            if self._forward_allowed:
                return States.FORWARD
            return self._current_state

        if self._left_emergency_turn and self._right_emergency_turn:
            if left > right:
                return States.LEFT
            return States.RIGHT
        if self._left_emergency_turn:
            return States.LEFT
        if self._right_emergency_turn:
            return States.RIGHT

        if self._forward_allowed:
            return States.FORWARD

        if self._current_state == States.FORWARD:
            if self._right_allowed and self._left_allowed:
                if left > right:
                    return States.LEFT
                return States.RIGHT

            if self._right_allowed:
                return States.RIGHT

            if self._left_allowed:
                return States.LEFT

        raise Exception("NO NEW STATE")

    def _get_new_state_real(
        self,
        left: float,
        right: float,
        forward_left: float | None,
        forward_right: float | None,
    ) -> States:
        if forward_left is None or forward_right is None:
            return self._current_state

        if self._current_state in [States.TURNAROUND, States.RIGHT, States.LEFT]:
            if self._forward_allowed:
                return States.FORWARD
            return self._current_state

        if self._forward_allowed:
            return States.FORWARD

        if self._forward_left_allowed and self._forward_right_allowed:
            if forward_left > forward_right:
                return States.FORWARD_LEFT
            return States.FORWARD_RIGHT
        if self._forward_left_allowed:
            return States.FORWARD_LEFT
        if self._forward_right_allowed:
            return States.FORWARD_RIGHT

        if self._current_state == States.FORWARD:
            if self._right_allowed and self._left_allowed:
                if left > right:
                    return States.LEFT
                return States.RIGHT

            if self._right_allowed:
                return States.RIGHT

            if self._left_allowed:
                return States.LEFT

        if not self._left_allowed and not self._right_allowed:
            return States.TURNAROUND
        return self._current_state

    def _error(self, points: numpy.typing.NDArray) -> float:
        if self._simulation:
            return self.SIM_OBJECTIVE_DISTANCE_WALL - float(np.median(points))
        return (
            np.clip(float(np.median(points)), -0.5, 0.5)
            - self.REAL_OBJECTIVE_DISTANCE_WALL
        )

    def _get_pd(self, error: float) -> float:
        if self._simulation:
            de: float = (error - float(self._prev_error)) / self._dt
            self._prev_error = error
            return self.SIM_KP * error + self.SIM_KD * de

        if self._prev_error is None:
            self._prev_error = error
        de = (error - float(self._prev_error)) / self._dt
        self._prev_error = error
        return -(self.REAL_KP * error + self.REAL_KD * de)

    def _apply_velocity_limits(
        self, v: float, w: float, w_limit_base_linear: float
    ) -> tuple[float, float]:
        w_max: float = (self.LINEAR_SPEED_MAX - abs(w_limit_base_linear)) / (
            self.TRACK / 2
        )
        w = float(np.clip(w, -w_max, w_max))
        v_best: float = self.LINEAR_SPEED_MAX - (abs(w) * (self.TRACK / 2))
        v = min(v, v_best)
        return v, w

    def compute_commands(
        self, z_scan: list[float], z_v: float, z_w: float
    ) -> tuple[float, float]:
        if self._simulation:
            return self._compute_commands_sim(z_scan)
        return self._compute_commands_real(z_scan)

    def _compute_commands_sim(self, z_scan: list[float]) -> tuple[float, float]:
        """Wall following exploration algorithm for simulation."""
        v: float = 0.0
        w: float = 0.0

        forward, left, right, forward_left, forward_right = self._get_lidar_areas(
            z_scan
        )

        if self._current_state not in [States.LEFT, States.RIGHT]:
            self._forward_allowed = min(forward) > self.SIM_ALLOWED_DISTANCE_FORWARD
            self._forward_turning = False
        else:
            self._forward_allowed = (
                min(forward) > self.SIM_ALLOWED_DISTANCE_FORWARD_IN_TURN
            )
            self._forward_turning = True

        self._left_allowed = min(left) > self.SIM_ALLOWED_DISTANCE_TURN
        self._right_allowed = min(right) > self.SIM_ALLOWED_DISTANCE_TURN

        if (
            all(forward < self.SIM_FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND)
            and all(left < self.SIM_FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND)
            and all(right < self.SIM_FORWARD_NO_EXIT_CORRIDOR_DISTANCE_TURNAROUND)
        ):
            self._current_state = States.TURNAROUND

        self._left_emergency_turn = (
            min(forward_right) < self.SIM_FORWARD_TURN_EMERGENCY_DISTANCE
        )
        self._right_emergency_turn = (
            min(forward_left) < self.SIM_FORWARD_TURN_EMERGENCY_DISTANCE
        )

        self._current_state = self._get_new_state(max(left), max(right))

        if min(forward) < self.SIM_EMERGENCY_DISTANCE_TURNAROUND:
            self._current_state = States.TURNAROUND

        if self._current_state == States.FORWARD:
            if not self._forward_turning:
                v = self.SIM_FORWARD_LINEAR_VELOCITY
            else:
                v = self.SIM_FORWARD_LINEAR_TURN_VELOCITY

            right_error: float = self._error(right[: self.SIM_SIDE_POINTS])
            w = self._get_pd(right_error)
        elif self._current_state == States.LEFT:
            v = self.SIM_TURN_LINEAR_VELOCITY
            w = self.SIM_TURN_ANGULAR_VELOCITY
        elif self._current_state == States.RIGHT:
            v = self.SIM_TURN_LINEAR_VELOCITY
            w = -self.SIM_TURN_ANGULAR_VELOCITY
        elif self._current_state == States.TURNAROUND:
            v = 0.0
            w = self.SIM_TURNAROUND_ANGULAR_VELOCITY

        self._prev_error = self._error(right[: self.SIM_SIDE_POINTS])

        # Keep the simulation branch aligned with the original sim controller.
        w_max: float = (self.LINEAR_SPEED_MAX - abs(v)) / (self.TRACK / 2)
        w = float(np.clip(w, -w_max, w_max))

        v_best: float = self.LINEAR_SPEED_MAX - (abs(w) * (self.TRACK / 2))
        v = min(v, v_best)

        return v, w

    def _compute_commands_real(self, z_scan: list[float]) -> tuple[float, float]:
        self.increment = 360 / len(z_scan) * np.pi / 180

        v: float = 0.0
        w: float = 0.0

        forward, left, right, forward_left, forward_right = self._get_lidar_areas(
            z_scan
        )

        if (
            len(forward) < 1
            or len(left) < 1
            or len(right) < 1
            or len(forward_left) < 1
            or len(forward_right) < 1
        ):
            return v, w

        if self._current_state not in [States.LEFT, States.RIGHT]:
            self._forward_allowed = (
                np.median(forward) > self.REAL_ALLOWED_DISTANCE_FORWARD
            )
            self._forward_turning = False
        else:
            self._forward_allowed = (
                max(forward) > self.REAL_ALLOWED_DISTANCE_FORWARD_IN_TURN
            )
            self._forward_turning = True

        self._forward_left_allowed = (
            np.median(forward_left) > self.REAL_ALLOWED_DISTANCE_FORWARD_TURN
        )
        self._forward_right_allowed = (
            np.median(forward_right) > self.REAL_ALLOWED_DISTANCE_FORWARD_TURN
        )

        self._left_allowed = np.median(left) > self.REAL_ALLOWED_DISTANCE_TURN
        self._right_allowed = np.median(right) > self.REAL_ALLOWED_DISTANCE_TURN

        self._current_state = self._get_new_state(
            max(left),
            max(right),
            np.median(forward_left),
            np.median(forward_right),
        )
        
        if self._current_state == States.FORWARD:
            if not self._forward_turning:
                v = self.REAL_FORWARD_LINEAR_VELOCITY
            else:
                v = self.REAL_FORWARD_LINEAR_TURN_VELOCITY

            right_error: float = self._error(right)
            w = self._get_pd(right_error)
        elif self._current_state == States.FORWARD_LEFT:
            self._prev_error = None
            v = self.REAL_FORWARD_LINEAR_TURN_VELOCITY
            w = self.REAL_FORWARD_TURNING_ANGULAR_VELOCITY
        elif self._current_state == States.FORWARD_RIGHT:
            self._prev_error = None
            v = self.REAL_FORWARD_LINEAR_TURN_VELOCITY
            w = -self.REAL_FORWARD_TURNING_ANGULAR_VELOCITY
        elif self._current_state == States.LEFT:
            self._prev_error = None
            v = self.REAL_TURN_LINEAR_VELOCITY
            w = self.REAL_TURN_ANGULAR_VELOCITY
        elif self._current_state == States.RIGHT:
            self._prev_error = None
            v = self.REAL_TURN_LINEAR_VELOCITY
            w = -self.REAL_TURN_ANGULAR_VELOCITY
        elif self._current_state == States.TURNAROUND:
            self._prev_error = None
            v = 0.0
            w = self.REAL_TURNAROUND_ANGULAR_VELOCITY

        w_base_linear = min(self.REAL_MINIMUM_LINEAR_VELOCITY, v)
        return self._apply_velocity_limits(v, w, w_base_linear)
