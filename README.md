# Decentralized UAV Swarms for Ground Target Protection

This ROS 2 package implements the experimental pipeline presented in **“Decentralized UAV Swarms for Ground Target Protection in GPS- and Communication-Denied Environments.”** A decentralized swarm of Crazyflie UAVs first encircles and protects a moving ground vehicle. When an attacking UAV is detected, the defenders transition toward it, estimate its motion from noisy relative measurements, encircle it, and collapse the formation if it enters the protected region.

Each defender estimates the target state and its angular separation from neighboring UAVs locally. The controller adapts the formation radius and angular velocity to the estimated target velocity. The algorithms therefore use relative observations rather than GPS or direct UAV-to-UAV communication. In the paper experiments, Vicon measurements with added Gaussian noise emulate onboard range-and-bearing sensing; low-level localization and target detection are outside the scope of this repository.

## Pipeline

The experiment has four operational stages:

1. **Ground-target protection:** the defenders take off and encircle the moving ground vehicle.
2. **Transition:** detection of an attacker causes the swarm to move toward it using flocking and collision avoidance.
3. **Attacker encirclement:** the defenders track and encircle the aerial target on an altitude-adaptive plane.
4. **Neutralization:** when the attacker enters the protected red zone, the encirclement radius collapses toward it.

The launch file starts one estimation-and-control pipeline and one relative-measurement emulator per defender, plus the attacker, ordering, Crazyflie server, watchdog, and motion-capture nodes.

## Main ROS 2 nodes

| Executable | Purpose |
| --- | --- |
| `pipeline_complete` | Runs on each defender. It contains the target-state and inter-agent phase filters, adaptive encirclement controller, flocking transition, mission state machine, and Crazyflie command generation. |
| `gps_scanner_ii` | Converts `/poses` into measurements expressed in each defender's initial/body-relative frame. It emulates the noisy relative sensor used by the algorithm and publishes the defender's latched initial pose. |
| `agents_order` | Determines the initial circular ordering of the defenders and continuously publishes the swarm order and distance to each leader. |
| `evader` | Controls the experimental attacking Crazyflie and publishes its detection state. |
| `motion_capture_tracking_node` | Third-party node that reads Vicon or another supported motion-capture system and publishes named rigid-body poses. |
| `crazyflie_server` / `watch_dog.py` | Third-party Crazyswarm2 nodes for Crazyflie communication, command forwarding, and safety monitoring. |

`filters.py` provides the invariant/unicycle target estimator and relative phase-difference filter used by `pipeline_complete`; it is a library module rather than a standalone node.

## Main topics and services

`{robot}` denotes a defender name from `crazyflies.yaml` (for example, `C01`).

| Name | Type | Producer → consumer | Description |
| --- | --- | --- | --- |
| `/poses` | `motion_capture_tracking_interfaces/NamedPoseArray` | motion capture → all experiment nodes | Ground-truth rigid-body poses. In this implementation they are transformed and noised to emulate relative sensing. |
| `/{robot}/gps_scanner_relative_poses` | `motion_capture_tracking_interfaces/NamedPoseArray` | `gps_scanner_ii` → `pipeline_complete` | Target and neighbor poses expressed relative to the defender. |
| `/{robot}/gps_scanner_global_poses` | `motion_capture_tracking_interfaces/NamedPoseArray` | `gps_scanner_ii` → monitoring | The same detected bodies expressed in the defender's initial frame. |
| `/{robot}/initial_pose` | `geometry_msgs/PoseStamped` | `gps_scanner_ii` → `pipeline_complete` | Latched initial pose used to define the local reference frame. |
| `/agents_order` | `crazyflie_interfaces/StringArray` | `agents_order` → defenders | Circular leader/follower ordering of the swarm. |
| `/{robot}/distance_to_leader` | `std_msgs/Float32` | `agents_order` → monitoring | Current Euclidean distance to the preceding defender. |
| `/encircle` | `std_msgs/Bool` | operator → defenders | Starts the encirclement mission. |
| `/landing` | `std_msgs/Bool` | operator/defenders → experiment nodes | Requests landing. |
| `/evade` | `std_msgs/Bool` | operator → `evader` | Starts the attacker's motion. |
| `/evader_detection` | `std_msgs/Bool` | attacker/defenders → defenders | Signals attacker detection and mission transitions. |
| `/{robot}/cmd_position` | `crazyflie_interfaces/Position` | controller → Crazyflie server | Position waypoint sent to a vehicle. |
| `/{robot}/cmd_velocity_world` | `crazyflie_interfaces/VelocityWorld` | `pipeline_complete` → Crazyflie server | World-frame velocity command used by the defender controller. |
| `/{robot}/relative/filtered/*` | `std_msgs/Float32` | filters/controller → monitoring | Estimated phase differences, angular velocity, radius, and radial correction. |
| `/{robot}/arm` | `crazyflie_interfaces/srv/Arm` | experiment nodes → Crazyflie server | Arms a vehicle. |
| `/{robot}/reboot` | `std_srvs/srv/Empty` | experiment nodes → Crazyflie server | Reboots a vehicle after landing. |

Additional filter-state topics are published below `/{robot}/unicycle/.../filtered/` using messages from `crazy_encirclement_interfaces`.

## Requirements

### Software

- Ubuntu 22.04 with ROS 2 Humble, or Ubuntu 24.04 with ROS 2 Jazzy.
- Python 3 and the Python packages `numpy`, `scipy`, `numpy-quaternion`, `PyYAML`, and `icecream`.
- `colcon`, `rosdep`, and the standard ROS 2 Python build tools.
- [Crazyswarm2](https://github.com/IMRCLab/crazyswarm2), which supplies `crazyflie`, `crazyflie_sim`, `crazyflie_interfaces`, the Crazyflie server, and watchdog. Follow its [official installation guide](https://imrclab.github.io/crazyswarm2/installation.html).
- [motion_capture_tracking](https://github.com/IMRCLab/motion_capture_tracking), which supplies the motion-capture node and `motion_capture_tracking_interfaces`. It supports Vicon, Qualisys, OptiTrack, VRPN, NOKOV, FZMotion, and Motion Analysis. It can also be installed as `ros-<DISTRO>-motion-capture-tracking` where available.
- [crazy_encirclement_interfaces](https://github.com/paaraujo/crazy_encirclement_interfaces/tree/master), the accompanying ROS 2 interface package containing `FilterUnicycleState` and `Metadata`. This package must be placed in the same workspace.

### Experimental hardware

The paper used three Crazyflies as defenders, one Crazyflie as the attacker, an AgileX Limo as the protected ground target, Crazyradio hardware, and a Vicon motion-capture system. Other platforms and range/bearing sensors can be used if they provide equivalent relative measurements and compatible command interfaces.

> **Safety:** Real multi-UAV experiments require a correctly calibrated motion-capture system, tested emergency-stop and landing procedures, sufficient flight volume, and appropriate physical protection. Validate configuration and controller gains in simulation before enabling motors.

## Installation

Create a ROS 2 workspace and clone all source dependencies into `src`:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

git clone --recursive https://github.com/IMRCLab/crazyswarm2.git
git clone --recursive https://github.com/IMRCLab/motion_capture_tracking.git
git clone --branch master https://github.com/paaraujo/crazy_encirclement_interfaces.git
git clone https://github.com/QUARRG/target_protection.git
```

Install dependencies and build:

```bash
cd ~/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install numpy scipy numpy-quaternion PyYAML icecream
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

For real Crazyflies, also configure the Crazyradio USB permissions and firmware as described by Crazyswarm2.

## Configuration and execution

Configure the enabled vehicles, radio addresses, initial positions, marker geometry, and motion-capture backend in Crazyswarm2's `crazyflie/config/crazyflies.yaml` and `motion_capture.yaml`. Every enabled vehicle must also have a `role` field used by this launch file:

```yaml
robots:
  C01:
    enabled: true
    role: pursuer
    # URI, type, initial_position, ...
  C23:
    enabled: true
    role: evader
    # URI, type, initial_position, ...
```

Filter, controller, noise, and loop-rate parameters are defined in [`config/filters.yaml`](config/filters.yaml). The values committed here correspond to the experimental pipeline and should be retuned for a different platform or sensing setup.

Launch the complete experiment with the C++ Crazyflie backend:

```bash
ros2 launch crazy_encirclement pipeline_complete_launch.py backend:=cpp mocap:=True rviz:=False
```

Alternative launch arguments include `backend:=cflib` and `backend:=sim`. The simulation backend additionally requires the Crazyswarm2 simulation dependencies and Crazyflie firmware Python bindings. Configuration files can be overridden explicitly:

```bash
ros2 launch crazy_encirclement pipeline_complete_launch.py \
  crazyflies_yaml_file:=/path/to/crazyflies.yaml \
  motion_capture_yaml_file:=/path/to/motion_capture.yaml
```

Mission commands can be sent from separate terminals:

```bash
ros2 topic pub --once /encircle std_msgs/msg/Bool '{data: true}'
ros2 topic pub --once /evade std_msgs/msg/Bool '{data: true}'
ros2 topic pub --once /landing std_msgs/msg/Bool '{data: true}'
```

## Citation

If you use this code, please cite:

```bibtex
@misc{silveria2026decentralizeduavswarmsground,
      title={Decentralized UAV Swarms for Ground Target Protection in GPS- and Communication-Denied Environments},
      author={Dimitria Silveria and Paulo Ricardo Marques de Araujo and Tiago Nascimento and Sidney Givigi},
      year={2026},
      eprint={2607.20710},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.20710},
}
```
