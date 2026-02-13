# FilterUnicycleState.msg
# Custom ROS message for FilterUnicycle state estimation
# State: [x, y, theta, omega, v, z_ground]

# Header with timestamp
std_msgs/Header header

# Position in the ground plane (x, y)
float64 x
float64 y

# Ground height estimate
float64 z_ground

# Heading angle (theta) in radians
float64 theta

# Angular velocity (omega) in rad/s
float64 angular_speed

# Linear velocity (v) in m/s
float64 linear_speed

# State covariance matrix (6x6 = 36 elements)
# Row-major order: [x, y, theta, omega, v, z_ground]
float64[36] covariance
