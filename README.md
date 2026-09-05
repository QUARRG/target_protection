# Decentralized UAV Swarms for Ground Target Protection

This ROS 2 package implements the experimental pipeline presented in **“Decentralized UAV Swarms for Ground Target Protection in GPS- and Communication-Denied Environments.”** A decentralized swarm of Crazyflie UAVs first encircles and protects a moving ground vehicle. When an attacking UAV is detected, the defenders transition toward it, estimate its motion from noisy relative measurements, encircle it, and collapse the formation if it enters the protected region.

Each defender estimates the target state and its angular separation from neighboring UAVs locally. The controller adapts the formation radius and angular velocity to the estimated target velocity. The algorithms therefore use relative observations rather than GPS or direct UAV-to-UAV communication. In the paper experiments, Vicon measurements with added Gaussian noise emulate onboard range-and-bearing sensing; low-level localization and target detection are outside the scope of this repository.

## Pipeline

The experiment has four operational stages:

1. **Ground-target protection:** the defenders take off and encircle the moving ground vehicle.
2. **Transition:** detection of an attacker causes the swarm to move toward it using flocking and collision avoidance.
3. **Attacker encirclement:** the defenders track and encircle the aerial target on an altitude-adaptive plane.
4. **Neutralization:** when the attacker enters the protected red zone, the encirclement radius collapses toward it.

The launch file starts one estimation-and-control pipeline and one relative-measurement emulator per defender, plus the attacker, ordering, Crazyflie server, and watchdog nodes. In hardware mode it obtains poses from motion capture. In simulation mode it instead starts CrazySim's MuJoCo/SITL environment and converts the simulated vehicle transforms to the same `/poses` interface used by the experiment nodes.

## Requirements

### Software

- Ubuntu 22.04 with ROS 2 Humble, or Ubuntu 24.04 with ROS 2 Jazzy.
- Python 3 and the Python packages `numpy`, `scipy`, `numpy-quaternion`, `PyYAML`, and `icecream`.
- `colcon`, `rosdep`, and the standard ROS 2 Python build tools.
- [Crazyswarm2](https://github.com/IMRCLab/crazyswarm2), which supplies `crazyflie`, `crazyflie_interfaces`, and the Crazyflie servers. The `crazyflie_sim` backend is **not** used by this simulation setup.
- The [project CrazySim fork](https://github.com/dimitriasilveria/CrazySim), including its Crazyflie firmware submodule. CrazySim supplies the MuJoCo physics/visualization process and runs one firmware-in-the-loop instance per drone. The project fork contains the LIMO integration used by this launch.
- [limo_ros2](https://github.com/agilexrobotics/limo_ros2), with this project's MuJoCo adaptation. This workspace uses `limo_description/mujoco/limo.xml`; the Gazebo plugins from the upstream LIMO packages are not involved in a CrazySim run.
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

git clone --recursive https://github.com/dimitriasilveria/crazyswarm2.git
git clone --recursive https://github.com/IMRCLab/motion_capture_tracking.git
git clone --recursive https://github.com/dimitriasilveria/CrazySim.git
git clone --branch humble https://github.com/agilexrobotics/limo_ros2.git
git clone https://github.com/dimitriasilveria/controller_pkg.git
git clone --branch master https://github.com/paaraujo/crazy_encirclement_interfaces.git
git clone https://github.com/QUARRG/target_protection.git
```

The upstream LIMO repository does not itself provide the custom `limo_description/mujoco/limo.xml` used here. Preserve the adapted `limo_ros2` checkout (including that model and its `limo_description/CMakeLists.txt` install rule), or retrieve those changes from the project branch before launching the simulation.

Keep the top-level Crazyswarm2 checkout as the ROS dependency. CrazySim also contains a Crazyswarm2 submodule, but the two copies must not be discovered, built, or sourced in the same ROS environment. This workspace keeps the complete CrazySim source tree out of `colcon` package discovery:

```bash
touch ~/ros2_ws/src/CrazySim/COLCON_IGNORE
```

Install the ROS dependencies and build the workspace with the system Python. Do not activate the CrazySim virtual environment for this build; ROS interface generation on Jazzy requires ROS's compatible EmPy installation rather than a venv package named `em`.

```bash
cd ~/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install numpy scipy numpy-quaternion PyYAML icecream
colcon build --symlink-install --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

Create the Python environment used by the CrazySim launcher and CFLib server. The launch file defaults to `~/venvs/crazyflie`:

```bash
python3 -m venv ~/venvs/crazyflie
source ~/venvs/crazyflie/bin/activate
python -m pip install --upgrade pip
git clone https://github.com/bitcraze/crazyflie-lib-python.git /tmp/crazyflie-lib-python
python -m pip install /tmp/crazyflie-lib-python
python -m pip install mujoco numpy Jinja2
deactivate
```

Install CFLib from its current source tree as shown above: CrazySim's UDP driver support may be newer than the released `cflib` package on PyPI.

Build the CrazySim SITL firmware once (and rebuild it after firmware changes):

```bash
cd ~/ros2_ws/src/CrazySim/crazyflie-firmware
mkdir -p sitl_make/build
cd sitl_make/build
cmake ..
cmake --build . -j4
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
ros2 launch target_protection pipeline_complete_launch.py backend:=cpp mocap:=True rviz:=False
```

The Python server can be selected on hardware with `backend:=cflib`.

### CrazySim simulation

Simulation uses the same experiment nodes and ROS command interfaces as hardware, but replaces the physical vehicles and motion-capture source as follows:

| Layer | Simulation implementation |
| --- | --- |
| Vehicle physics and visualization | CrazySim with MuJoCo |
| Flight controller | Crazyflie firmware running in SITL, one process per vehicle |
| ROS-to-firmware connection | Crazyswarm2 `crazyflie_server_py` with `backend:=cflib` over UDP |
| Drone poses | Firmware pose logging → Crazyflie server TF → `sim_pose_bridge` → `/poses` |
| Ground vehicle | Kinematic LIMO MuJoCo model driven by `sim_point_mass` |

Run the complete simulation with:

```bash
ros2 launch target_protection pipeline_complete_launch.py use_sim:=True
```

`use_sim:=True` forces `backend:=cflib`, disables motion capture, enables RViz, and starts CrazySim by default. It does **not** select Crazyswarm2's `backend:=sim` or launch the `crazyflie_sim` server. The CrazySim MuJoCo window and RViz are therefore shown together.

The enabled robots in `crazyflies.yaml` must use consecutive UDP ports in their YAML order, beginning at port 19850. For four enabled robots, use `udp://127.0.0.1:19850` through `udp://127.0.0.1:19853`. The launch validates this mapping because CrazySim assigns these ports by spawn index.

At startup, the launch generates a randomized layout and writes the exact drone coordinates to `/tmp/target_protection_crazysim_layout.txt` before starting MuJoCo:

- The LIMO starts at a random nonzero point inside the 2 m by 2 m square centered at the origin.
- Pursuers start around the LIMO with independently randomized phase and radius between `0.5 * radius_nominal` and `2.0 * radius_nominal`.
- The evader starts at `evader_initial_distance` from the LIMO. Its default is 10 m and values below 4 m are rejected.
- `simulation_layout_seed:=-1` produces a new random layout. Supply a nonnegative seed for a reproducible layout.

The default CrazySim vehicle model is `cf21B_500`, the Crazyflie 2.1 Brushless model. Useful launch overrides include:

```bash
ros2 launch target_protection pipeline_complete_launch.py \
  use_sim:=True \
  simulation_layout_seed:=42 \
  evader_initial_distance:=6.0 \
  crazysim_model:=cf21B_500
```

If the workspace, virtual environment, firmware, or LIMO model is installed elsewhere, override `crazysim_firmware_path`, `cflib_pythonpath`, `crazysim_python_bin`, or `crazysim_limo_model_path`. Set `start_crazysim:=False` only when attaching this launch to an already-running compatible CrazySim instance.

Simulation experiment events are published automatically according to
`config/simulation_experiment.yaml`. The configured times are measured in
seconds from the simulation experiment controller's startup:

- `defenders_takeoff_time` publishes `True` on `/defenders_takeoff`.
- `evader_takeoff_time` publishes `True` on `/evader_takeoff`.
- `encirclement_time` publishes `True` on `/encircle`.
- `start_limo_time` publishes `True` on `/start_limo`. The LIMO then follows a 2 m-radius circular path; the launch currently configures an angular velocity of 0.2 rad/s.
- `evade_time` publishes `True` on `/evade`.
- `land_time` publishes `True` on `/landing`.
- When `experiment_type` is `give up`, `evader_desengage` publishes `False`
  on `/evade`.

Configuration files can be overridden explicitly:

```bash
ros2 launch target_protection pipeline_complete_launch.py \
  crazyflies_yaml_file:=/path/to/crazyflies.yaml \
  motion_capture_yaml_file:=/path/to/motion_capture.yaml
```

A different experiment schedule can be selected with:

```bash
ros2 launch target_protection pipeline_complete_launch.py \
  use_sim:=True \
  simulation_experiment_file:=/path/to/simulation_experiment.yaml
```

Mission commands can be sent from separate terminals:

```bash
ros2 topic pub --once /encircle std_msgs/msg/Bool '{data: true}'
ros2 topic pub --once /start_limo std_msgs/msg/Bool '{data: true}'
ros2 topic pub --once /evade std_msgs/msg/Bool '{data: true}'
ros2 topic pub --once /landing std_msgs/msg/Bool '{data: true}'
```

For a quick state-path check during simulation:

```bash
ros2 topic hz /tf
ros2 topic echo /poses --once
ros2 topic echo /LIMO/pose --once
```

`/poses` is deliberately withheld until `sim_pose_bridge` has received a direct `world` TF for every enabled drone and for `LIMO`. A repeated `"/poses is waiting for TF frames"` warning therefore means the corresponding SITL instance or its firmware pose log is not reaching the CFLib server; it is not a motion-capture configuration problem.

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
