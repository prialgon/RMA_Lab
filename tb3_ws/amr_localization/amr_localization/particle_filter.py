import datetime
import math
import numpy as np
import os
import pytz
import random

from amr_localization.maps import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        particle_count: int,
        sigma_v: float = 0.05,
        sigma_w: float = 0.1,
        sigma_z: float = 0.2,
        sensor_range_max: float = 8.0,
        sensor_range_min: float = 0.16,
        global_localization: bool = True,
        initial_pose: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        initial_pose_sigma: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        logger=None,
        simulation: bool = False,
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].
            sensor_range_max: Maximum sensor measurement range [m].
            sensor_range_min: Minimum sensor measurement range [m].
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].
            logger: Logger object to output messages with different severity levels.
            simulation: True if running in simulation, False if running on the real robot.

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._logger = logger
        self._particle_count: int = particle_count
        self._sensor_range_max: float = sensor_range_max
        self._sensor_range_min: float = sensor_range_min
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._simulation: bool = simulation
        self._iteration: int = 0

        self._map: Map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        self._particles = self._init_particles(
            particle_count, global_localization, initial_pose, initial_pose_sigma
        )
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self._n_prev_clusters = particle_count

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        # TODO: 3.10. Complete the missing function body with your code.
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
        
        particles_with_approx = np.column_stack((self._particles[:, :2], np.sin(self._particles[:, 2]), np.cos(self._particles[:, 2])))
        coords = self._particles[:, :2]
        clusters = DBSCAN(eps=0.2).fit(coords)

        labels = set(clusters.labels_)
        if -1 in labels:
            labels.remove(-1)
        
        
        if len(labels) == 1:
            
            pose_calculation = particles_with_approx[clusters.labels_ == clusters.labels_[0]].mean(axis=0)
            theta_mean = np.arctan2(pose_calculation[2], pose_calculation[3])
            pose = (pose_calculation[0], pose_calculation[1], theta_mean)
            
            if len(self._particles) > 50:
                self._particles = self._init_particles(50, False, pose, (0.05, 0.05, math.radians(5)))
            localized = True
            self._particle_count = 50
        else:
            if len(set(clusters.labels_)) < self._n_prev_clusters:
                self._particle_count = max(50, int(self._particle_count * 0.8))
            
            max_count_cluster = np.argmax(np.bincount(clusters.labels_[clusters.labels_ != -1]))
            pose_calculation = particles_with_approx[clusters.labels_ == max_count_cluster].mean(axis=0)
            theta_mean = np.arctan2(pose_calculation[2], pose_calculation[3])
            pose = (pose_calculation[0], pose_calculation[1], theta_mean)
            self._n_prev_clusters = len(set(clusters.labels_))
            
        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        old_positions = self._particles[:, [0, 1]].copy()
        
        num_particles = self._particles.shape[0]
        # self._logger.warn(f"ERR: {w}")
        noise_v = np.random.normal(loc=0, scale=self._sigma_v, size=num_particles)
        noise_w = np.random.normal(loc=-0.4, scale=self._sigma_w, size=num_particles)

        delta_position = (v + noise_v) * self._dt
        delta_rotation = (w + noise_w) * self._dt

        orientations = self._particles[:, 2]

        delta_x = np.cos(orientations + delta_rotation / 2.0) * delta_position
        delta_y = np.sin(orientations + delta_rotation / 2.0) * delta_position

        self._particles[:, 0] += delta_x
        self._particles[:, 1] += delta_y
        self._particles[:, 2] += delta_rotation

        self._particles[:, 2] = np.mod(self._particles[:, 2], 2 * np.pi)

        for idx, particle in enumerate(self._particles):
            segment = [old_positions[idx], particle[[0, 1]]]
            intersection, _ = self._map.check_collision(segment, compute_distance=False)

            if intersection:
                self._particles[idx, [0, 1]] = intersection

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).
        
        weights: list[float] = []
        
        for particle in self._particles:
            w: float = self._measurement_probability(measurements, particle)
            weights.append(w)
            
        bins = np.cumsum(weights)
        
        W: float = np.sum(weights)
        self._logger.warn(f"W: {W}; Count: {self._particle_count}")
        u_0: float = np.random.uniform(low=0, high=W/self._particle_count)
        
        u = u_0 + np.arange(self._particle_count) * W/self._particle_count
            
        idx = np.digitize(u, bins)
        
        self._particles = self._particles[idx]
            

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(
        self,
        particle_count: int,
        global_localization: bool,
        initial_pose: tuple[float, float, float],
        initial_pose_sigma: tuple[float, float, float],
    ) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=float)

        # TODO: 3.4. Complete the missing function body with your code.
        valid_particles = []
        x_min, y_min, x_max, y_max = self._map.bounds()
        while len(valid_particles) < particle_count:
            # count missing particles
            needed = particle_count - len(valid_particles)

            # oversample not to run this loop many times
            batch_size = int(needed * 2.0)

            if global_localization:
                x_cand = np.random.uniform(x_min, x_max, size=batch_size)
                y_cand = np.random.uniform(y_min, y_max, size=batch_size)
            else:
                # random (normal) noise
                x_cand = np.random.normal(initial_pose[0], initial_pose_sigma[0], size=batch_size)
                y_cand = np.random.normal(initial_pose[1], initial_pose_sigma[1], size=batch_size)

            # verify the particles are valid
            for x, y in zip(x_cand, y_cand):
                if self._map.contains((x, y)):
                    valid_particles.append((x, y))
                    if len(valid_particles) == particle_count:
                        break

        particles_xy = np.array(valid_particles)

        particles[:, 0] = particles_xy[:, 0]
        particles[:, 1] = particles_xy[:, 1]

        # add theta
        particles[:, 2] = np.random.choice(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2], size=particle_count
        )

        return particles

    def _sense(self, pose: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every LiDAR ray given the robot's pose.

        Args:
            pose: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; nan if a sensor is out of range.

        """
        z_hat: list[float] = []

        # TODO: 3.6. Complete the missing function body with your code.
        n: int = 16
        rays = self._lidar_rays(pose, np.arange(n).tolist(), degree_increment=360 / n)

        for ray in rays:
            z_hat.append(self._map.check_collision(ray, compute_distance=True)[1])

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 3.7. Complete the function body (i.e., replace the code below).
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def _lidar_rays(
        self, pose: tuple[float, float, float], indices: tuple[float], degree_increment: float = 1.5
    ) -> list[list[tuple[float, float]]]:
        """Determines the simulated LiDAR ray segments for a given robot pose.

        Args:
            pose: Robot pose (x, y, theta) in [m] and [rad].
            indices: Rays of interest in counterclockwise order (0 for to the forward-facing ray).
            degree_increment: Angle difference of the sensor between contiguous rays [degrees].

        Returns: Ray segments. Format:
                 [[(x0_start, y0_start), (x0_end, y0_end)],
                  [(x1_start, y1_start), (x1_end, y1_end)],
                  ...]

        """
        x, y, theta = pose

        # Convert the sensor origin to world coordinates
        x_start = x - 0.035 * math.cos(theta)
        y_start = y - 0.035 * math.sin(theta)

        rays = []

        for index in indices:
            ray_angle = math.radians(degree_increment * index)
            x_end = x_start + self._sensor_range_max * math.cos(theta + ray_angle)
            y_end = y_start + self._sensor_range_max * math.sin(theta + ray_angle)
            rays.append([(x_start, y_start), (x_end, y_end)])

        return rays

    def _measurement_probability(
        self, measurements: list[float], particle: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        the minimum sensor range to perform the computation because the environment is smaller
        than the maximum range.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        probability = 1.0

        # TODO: 3.8. Complete the missing function body with your code.

        n: int = 16

        measurements = np.nan_to_num(measurements, nan=self._sensor_range_min, posinf=self._sensor_range_max, neginf=self._sensor_range_min)

        z_hat: list[float] = self._sense(particle)
        z_hat = np.nan_to_num(
            z_hat, 
            nan=self._sensor_range_max,
            posinf=self._sensor_range_max, 
            neginf=self._sensor_range_min
        )

        step = len(measurements) // n
        downsampled_measurements = [measurements[i] for i in range(0, n * step, step)]

        for z_sensed, z_real in zip(z_hat, downsampled_measurements):
            probability *= self._gaussian(z_sensed, self._sigma_z, z_real)

        return probability
