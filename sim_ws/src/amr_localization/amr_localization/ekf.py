import datetime
import math
import numpy as np
import os
import pytz
import random

from amr_localization.maps import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class EKF:
    """Extended Kalman Filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensor_range_max: float = 8.0,
        sensor_range_min: float = 0.16,
        initial_pose: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        initial_pose_sigma: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        logger=None,
        simulation: bool = False,
    ):
        """EKF class initializer.

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
        self._logger = logger
        self._sensor_range_max: float = sensor_range_max
        self._sensor_range_min: float = sensor_range_min
        self._simulation: bool = simulation
        self._iteration: int = 0
        self._pose = None
        self._sigma = None

        
        sigma_x = 0.01      
        sigma_y = 0.01     
        sigma_theta = 0.02  

        self._Rt = np.diag([sigma_x**2, sigma_y**2, sigma_theta**2])
        
        
        sigma_lidar = 0.1

        self._Qt = np.eye(16) * (sigma_lidar**2)

        self._map: Map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        self._particles = self._init_vars(
            initial_pose, initial_pose_sigma
        )
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )


    def motion_model(self, x, u):
        """
        Modelo de movimiento g(x, u)
        x: Vector de estado [x, y, theta]
        u: Vector de control [v, w] (velocidad lineal, velocidad angular)
        dt: Paso de tiempo (delta t)
        """
        x_prev, y_prev, theta_prev = x
        v, w = u

        # Umbral pequeño para evitar división por cero
        if abs(w) < 1e-9:
            # Movimiento en línea recta (Modelo lineal)
            x_next = x_prev + v * np.cos(theta_prev) * self._dt
            y_next = y_prev + v * np.sin(theta_prev) * self._dt
            theta_next = theta_prev
        else:
            # Modelo circular (el de la imagen)
            ratio = v / w
            x_next = x_prev - ratio * np.sin(theta_prev) + ratio * np.sin(theta_prev + w * self._dt)
            y_next = y_prev + ratio * np.cos(theta_prev) - ratio * np.cos(theta_prev + w * self._dt)
            theta_next = theta_prev + w * self._dt

        # Es recomendable normalizar el ángulo theta entre -pi y pi
        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_next, y_next, theta_next])
    
    def jacobian_motion(self, x, u):
        """
        Calcula el Jacobiano Gt del modelo de movimiento circular.
        x: Vector de estado [x, y, theta]
        u: Vector de control [v, w]
        dt: Paso de tiempo
        """
        theta_prev = x[2]
        v, w = u
        
        # Inicializamos la matriz identidad 3x3
        G = np.eye(3)
        
        if abs(w) < 1e-9:
            # Caso movimiento rectilíneo (Límite cuando w -> 0)
            G[0, 2] = -v * np.sin(theta_prev) * self._dt
            G[1, 2] =  v * np.cos(theta_prev) * self._dt
        else:
            # Caso general (Fórmula de la imagen)
            ratio = v / w
            G[0, 2] = -ratio * np.cos(theta_prev) + ratio * np.cos(theta_prev + w * self._dt)
            G[1, 2] = -ratio * np.sin(theta_prev) + ratio * np.sin(theta_prev + w * self._dt)
            
        return G


    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1
        n_mu = self.motion_model(self._pose, np.array([v, w]))
        G = self.jacobian_motion(self._pose, np.array([v, w]))
        n_sigma = G @ self._sigma @ G.T + self._Rt

        return n_mu, n_sigma   


    def jacobian_observation(self, pose, z_hat, epsilon=1e-6):
        """
        Calcula el Jacobiano H de la observación mediante diferencias finitas.
        pose: Vector [x, y, theta] actual (predicho).
        epsilon: El tamaño de la pequeña perturbación.
        """
       
        z_nom = z_hat
        num_rays = len(z_nom)

        H = np.zeros((num_rays, 3))

        pose_x = pose + np.array([epsilon, 0, 0])
        z_x = self._sense(pose_x)
        H[:, 0] = (z_x - z_nom) / epsilon

        pose_y = pose + np.array([0, epsilon, 0])
        z_y = self._sense(pose_y)
        H[:, 1] = (z_y - z_nom) / epsilon

        pose_theta = pose + np.array([0, 0, epsilon])
        z_theta = self._sense(pose_theta)
        H[:, 2] = (z_theta - z_nom) / epsilon

        return H
    

    def correct(self, n_mu, n_sigma, measurements):
        n = 16

        # Clean measurements
        measurements = np.nan_to_num(measurements, nan=self._sensor_range_min, posinf=self._sensor_range_max, neginf=self._sensor_range_min)

        step = len(measurements) // n
        downsampled_measurements = np.array([measurements[i] for i in range(0, n * step, step)])

        z_hat = self._sense(n_mu)
        H = self.jacobian_observation(n_mu, z_hat)

        St = H @ n_sigma @ H.T + self._Qt
        Kt = n_sigma @ H.T @ np.linalg.inv(St)

        innovation = downsampled_measurements - z_hat

        mu = n_mu + Kt @ innovation

        mu[2] = (mu[2] + np.pi) % (2 * np.pi) - np.pi

        I = np.eye(len(mu))
        sigma = (I - Kt @ H) @ n_sigma

        self._pose = mu
        self._sigma = sigma


    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """

        if orientation:
            dx = [math.cos(self._pose[2])]
            dy = [math.sin(self._pose[2])]
            axes.quiver(
                self._pose[0],
                self._pose[1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._pose[0], self._pose[1], "bo", markersize=1)

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

    def _init_vars(self, initial_pose, initial_pose_sigma):
        self._pose = np.array(initial_pose, dtype=float)
        sigma = np.array(initial_pose_sigma, dtype=float)
        if sigma.shape == (3,):
            sigma = np.diag(np.square(sigma))
        self._sigma = sigma

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
            dist = self._map.check_collision(ray, compute_distance=True)[1]
            if math.isnan(dist) or dist > self._sensor_range_max:
                dist = self._sensor_range_max
            elif dist < self._sensor_range_min:
                dist = self._sensor_range_min
            z_hat.append(dist)

        return np.array(z_hat)

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
