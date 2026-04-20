import math


class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(
        self,
        dt: float,
        lookahead_distance: float = 0.5,
        goal_tolerance: float = 0.05,
        logger=None,
        simulation: bool = False,
    ):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].
            goal_tolerance: Distance to the final path point considered reached [m].
            logger: Logger object to output messages with different severity levels.
            simulation: True if running in simulation, False if running on the real robot.

        """
        self._dt: float = dt
        self._logger = logger
        self._lookahead_distance: float = lookahead_distance
        self._goal_tolerance: float = goal_tolerance
        self._path: list[tuple[float, float]] = []
        self._simulation: bool = simulation

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        
        if len(self.path) == 0:
            return 0.0, 0.0

        dist_to_goal = math.dist((x, y), self.path[-1])
        if dist_to_goal < self._goal_tolerance:
            return 0.0, 0.0

        _, current_idx = self._find_closest_point(x, y)
        target_point = self._find_target_point((x, y), current_idx)

        lookahead_distance = max(math.dist((x, y), target_point), 1e-6)
        beta = math.atan2(target_point[1] - y, target_point[0] - x)
        alpha = (beta - theta + math.pi) % (2 * math.pi) - math.pi
        curvature = 2.0 * math.sin(alpha) / lookahead_distance

        max_linear_speed = 0.22 if self._simulation else 0.16
        min_linear_speed = 0.08 if self._simulation else 0.05
        max_angular_speed = 2.0 if self._simulation else 1.5

        if abs(alpha) > math.pi / 4:
            return 0.0, math.copysign(max_angular_speed, alpha)

        heading_scale = max(0.25, 1.0 - abs(alpha) / (math.pi / 2))
        v = max_linear_speed * heading_scale

        # if dist_to_goal < self._lookahead_distance:
        #     v *= max(0.35, dist_to_goal / max(self._lookahead_distance, 1e-6))

        v = min(max_linear_speed, max(min_linear_speed, v))
        w = max(-max_angular_speed, min(max_angular_speed, curvature * v))
        
        return v, w

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.9. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_xy = (0.0, 0.0)
        closest_idx = 0
        min_distance: float = math.inf
        for idx, node in enumerate(self.path):
            distance: float = math.dist(node, (x, y))
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
                closest_xy = node

        return closest_xy, closest_idx

    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance."""
        for node_idx in range(origin_idx, len(self.path)):
            node = self.path[node_idx]
            if math.dist(node, origin_xy) > self._lookahead_distance:
                return node

        return self.path[-1]
