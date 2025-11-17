# Crazy Encirclement System Architecture Diagram

## 1. Topic Communication Details

### 1.1 Global Topics

| Topic | Type | Publisher | Subscriber(s) | Description |
|-------|------|-----------|---------------|-------------|
| `/poses` | NamedPoseArray | motion_capture_tracking_node | agents_order, gps_node | Raw Vicon pose data for all robots |
| `/agents_order` | StringArray | agents_order | circle_distortion (all) | Ordered list of agents by initial phase |
| `/landing` | Bool | User/CLI | circle_distortion (all) | Command to initiate landing |
| `/encircle` | Bool | User/CLI | circle_distortion (all) | Command to start encirclement |

### 1.2 Per-Robot Topics (Replace `{robot}` with C01, C20, etc.)

| Topic | Type | Publisher | Subscriber(s) | Description |
|-------|------|-----------|---------------|-------------|
| `/{robot}/vicon_position` | Position | gps_node | circle_distortion | Clean Vicon position |
| `/{robot}/gps_position` | Position | gps_node | circle_distortion | Noisy GPS-like position |
| `/{robot}/cmd_position` | Position | circle_distortion | crazyflie_server | Position commands to robot |
| `/{robot}/filtered/phase` | Float32 | FilterGPS (via circle_distortion) | circle_distortion (leader & follower) | Filtered phase angle |
| `/{robot}/filtered/pose` | PoseStamped | FilterGPS (via circle_distortion) | Internal use | Filtered pose estimate |
| `/{robot}/filtered/omega` | Float32 | circle_distortion | Internal monitoring | Phase velocity (omega) |
| `/{robot}/distance_to_leader` | Float32 | agents_order | Monitoring | Distance to preceding agent |
| `/{robot}/reboot` | Empty (Service) | circle_distortion | crazyflie_server | Reboot service call |

## 2. Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAUNCH SEQUENCE                              │
│  1. Parse YAML configs (crazyflies, motion_capture, filters)        │
│  2. Start motion_capture_tracking_node                              │
│  3. Start crazyflie_server                                          │
│  4. Start agents_order node                                         │
│  5. For each robot: Start GPS, circle_distortion, watch_dog         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     INITIALIZATION PHASE                            │
│  • Motion capture publishes /poses                                  │
│  • GPS node starts publishing vicon & gps positions                 │
│  • agents_order computes initial phases & publishes order           │
│  • circle_distortion receives order, identifies leader/follower     │
│  • FilterGPS initialized with parameters                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      STATE 0: TAKEOFF                               │
│  • Receive initial pose from vicon                                  │
│  • Generate takeoff trajectory                                      │
│  • Command positions to reach hover_height (0.9m)                   │
│  • Duration: ~4 seconds                                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       STATE 1: HOVER                                │
│  • Hold position at initial (x, y, hover_height)                    │
│  • Wait for /encircle command                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ /encircle command
┌─────────────────────────────────────────────────────────────────────┐
│                   STATE 2: ENCIRCLEMENT                             │
│                                                                     │
│  PREDICTION CYCLE (Faster):                                         │
│    1. Read leader & follower phases                                 │
│    2. Compute omega via phase_controller                            │
│    3. filter.predict(omega, dt)                                     │
│    4. Publish filtered phase & omega                                │
│    5. Send cmd_position to crazyflie                                │
│                                                                     │
│  UPDATE CYCLE (Slower):                                             │
│    1. Receive GPS measurement                                       │
│    2. filter.update(measurement)                                    │
│    3. Publish corrected filtered phase                              │
│    4. Send updated cmd_position                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ /landing command
┌─────────────────────────────────────────────────────────────────────┐
│                      STATE 3: LANDING                               │
│  • Capture final pose                                               │
│  • Generate landing trajectory                                      │
│  • Descend to ground (~2 seconds)                                   │
│  • Call reboot service                                              │
│  • Shutdown node                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Error Handling & Safety

- **watch_dog**: Monitors robot status and triggers emergency procedures
- **Reboot service**: Called after landing to safely reset the system
- **QoS Settings**: BEST_EFFORT reliability for motion capture data
- **Timeout handling**: `rclpy.spin_once()` with timeouts during initialization
- **State machine**: Prevents invalid state transitions
- **Filter stability**: Covariance symmetrization and orthonormalization

## 4. Key Design Patterns

1. **Distributed Phase Synchronization**: Each agent maintains its own phase and subscribes to neighbors
2. **Lie Group Filtering**: FilterGPS operates on SO(3) manifold for rotation estimates
3. **Dual-rate Processing**: High-frequency prediction and low-frequency updates
4. **Leader-Follower Topology**: Ring formation with each agent following predecessor
5. **GPS Noise Simulation**: gps_node adds Gaussian noise to simulate realistic measurements
