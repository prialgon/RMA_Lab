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
        # TODO: 4.11. Complete the function body with your code (i.e., compute v and w).
        v = 0.08
        w = 0.0

        if len(self.path) == 0:
            return 0.0, 0.0

        dist_to_goal = math.dist((x, y), self.path[-1])
        if dist_to_goal < self._goal_tolerance:
            return 0.0, 0.0

        current_point, current_idx = self._find_closest_point(x, y)
        target_point = self._find_target_point((x, y), current_idx)

        l: float = math.dist((x, y), target_point)
        beta: float = math.atan2((target_point[1] - y), (target_point[0] - x))
        alpha: float = beta - theta
        
        # Normalize alpha between -pi and pi
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        if l > 1e-4:
            w = 2 * v * math.sin(alpha) / l
        else:
            w = 0.0

        if abs(alpha) > math.pi / 4:
            v = 0.0
            w = math.copysign(1.8, alpha)

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
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.10. Complete the function body with your code (i.e., determine target_xy).
        # target_xy = None

        # candidates = []
        # for node_idx in range(origin_idx, len(self.path)):
        #     if math.dist(self.path[node_idx], origin_xy) > self._lookahead_distance:
        #         candidates.append((self.path[node_idx], math.dist(self.path[node_idx], origin_xy)))

        # if len(candidates) == 0:
        #     target_xy = self.path[-1]
        # else:
        #     target_xy = min(candidates, key=lambda x: x[1])[0]

        # return target_xy
        
        if len(self.path) < 2:
            return self.path[-1] if self.path else origin_xy

        L = self._lookahead_distance
        x_r, y_r = origin_xy

        for i in range(origin_idx, len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]

            # Parametrize the segment as P(t) = P1 + t * (P2 - P1), t in [0, 1].
            # Solve |P(t) - robot|^2 = L^2 -> quadratic a*t^2 + b*t + c = 0.
            dx = x2 - x1
            dy = y2 - y1
            fx = x1 - x_r
            fy = y1 - y_r

            a = dx * dx + dy * dy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - L * L

            # Degenerate segment or no real intersection with this segment.
            if a < 1e-12:
                continue
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue

            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # Prefer the exit intersection (largest t in [0, 1]): that is the
            # point where the path leaves the lookahead circle moving forward.
            for t in (t2, t1):
                if 0.0 <= t <= 1.0:
                    return (x1 + t * dx, y1 + t * dy)

        # No segment intersects the lookahead circle: aim at the final goal.
        return self.path[-1]
