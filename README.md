# RMA Lab — Autonomous Mobile Robot

ROS 2 workspace for the Autonomous Mobile Robots (RMA) course at ICAI. A TurtleBot3 Burger explores an unknown map, localizes itself, plans a path, and drives to a goal — both in CoppeliaSim and on the real robot.

## Workspaces

| Folder       | Target                                        |
| ------------ | --------------------------------------------- |
| `sim_ws/`    | Simulation in CoppeliaSim                     |
| `tb3_ws/`    | Real TurtleBot3 Burger                        |
| `ssh_config` | SSH config for connecting to the robot        |

Both workspaces share the same architecture. Only parameter tuning and a few hardware-specific nodes differ.

## Pipeline

```
Wall follower  →  Particle filter  →  PRM planner  →  Pure pursuit  →  /cmd_vel
 (explore)        (localize)          (path to goal)   (track path)
```

1. **Wall follower** — safely explores the environment while the pose is unknown.
2. **Particle filter** (Python + C++ port) — global localization against a known map using LiDAR.
3. **EKF** — pose tracking once the particle filter has converged.
4. **PRM + A\*** — samples a roadmap over the free space and finds a smoothed path to the goal.
5. **Pure pursuit** — follows the path and publishes `/cmd_vel` velocity commands.

A **lifecycle manager** brings all nodes up in the right order.

## ROS 2 packages

- `amr_bringup` — launch files and lifecycle manager
- `amr_control` — wall follower + pure pursuit
- `amr_localization` — particle filter + EKF (Python)
- `amr_localization_cpp` — particle filter (C++ via nanobind)
- `amr_planning` — PRM + A\* path planner
- `amr_simulation` — CoppeliaSim bridge (sim_ws only)
- `amr_turtlebot3` — odometry and monitoring for the real robot (tb3_ws only)
- `amr_msgs` — custom messages (`PoseStamped`, `AllowMove`)

## Build & run (simulation)

```bash
cd sim_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch amr_bringup project.launch.py
```

Start CoppeliaSim and open a scene from `sim_ws/src/amr_simulation/scenes/` before launching.

## Build & run (real robot)

```bash
cd tb3_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch amr_bringup project.launch.py
```

SSH into the TurtleBot3 using `ssh_config` to bring up its own drivers.

## Requirements

- Ubuntu 22.04 + ROS 2 Humble
- CoppeliaSim 4.5+ (simulation only)
- Python: `numpy`, `matplotlib`, `scikit-learn`, `transforms3d`, `pytz`
- C++ build: `colcon`, `cmake`, `nanobind`

## Labs

Branches `feature/p2`, `feature/p3`, `feature/p4` hold the incremental lab deliverables. `feature/project` is the final integrated project.
