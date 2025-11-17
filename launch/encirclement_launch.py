import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.conditions import LaunchConfigurationEquals
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def parse_yaml(context):
    # Load the crazyflies YAML file
    crazyflies_yaml = LaunchConfiguration('crazyflies_yaml_file').perform(context)
    with open(crazyflies_yaml, 'r') as file:
        crazyflies = yaml.safe_load(file)

    # server params
    server_yaml = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'server.yaml')

    with open(server_yaml, 'r') as ymlfile:
        server_yaml_content = yaml.safe_load(ymlfile)

    server_params = [crazyflies] + [server_yaml_content['/crazyflie_server']['ros__parameters']]
    # robot description
    urdf = os.path.join(
        get_package_share_directory('crazyflie'),
        'urdf',
        'crazyflie_description.urdf')
    
    with open(urdf, 'r') as f:
        robot_desc = f.read()

    server_params[1]['robot_description'] = robot_desc

    # construct motion_capture_configuration
    motion_capture_yaml = LaunchConfiguration('motion_capture_yaml_file').perform(context)
    with open(motion_capture_yaml, 'r') as ymlfile:
        motion_capture_content = yaml.safe_load(ymlfile)
    motion_capture_params = motion_capture_content['/motion_capture_tracking']['ros__parameters']
    motion_capture_params['rigid_bodies'] = dict()
    for key, value in crazyflies['robots'].items():
        type = crazyflies['robot_types'][value['type']]
        if value['enabled'] and type['motion_capture']['enabled']:
            motion_capture_params['rigid_bodies'][key] =  {
                    'initial_position': value['initial_position'],
                    'marker': type['motion_capture']['marker'],
                    'dynamics': type['motion_capture']['dynamics'],
                }
    # copy relevent settings to server params
    server_params[1]['poses_qos_deadline'] = motion_capture_params['topics']['poses']['qos']['deadline']
    Nodes = []
    Nodes.append(Node(
            package='motion_capture_tracking',
            executable='motion_capture_tracking_node',
            condition=IfCondition(PythonExpression(["'", LaunchConfiguration('backend'), "' != 'sim' and '", LaunchConfiguration('mocap'), "' == 'True'"])),
            name='motion_capture_tracking',
            output='screen',
            parameters= [motion_capture_params],
        ))
    Nodes.append(Node(
            package='crazyflie',
            executable='crazyflie_server.py',
            condition=LaunchConfigurationEquals('backend','cflib'),
            name='crazyflie_server',
            output='screen',
            parameters= server_params,
        ))
    Nodes.append(Node(
            package='crazyflie',
            executable='crazyflie_server',
            condition=LaunchConfigurationEquals('backend','cpp'),
            name='crazyflie_server',
            output='screen',
            parameters= server_params,
            prefix=PythonExpression(['"xterm -e gdb -ex run --args" if ', LaunchConfiguration('debug'), ' else ""']),
        ))
    Nodes.append(Node(
            package='crazyflie_sim',
            executable='crazyflie_server',
            condition=LaunchConfigurationEquals('backend','sim'),
            name='crazyflie_server',
            output='screen',
            emulate_tty=True,
            parameters= server_params,
        ))

    # Filter configuration
    filter_yaml = os.path.join(
        get_package_share_directory('crazy_encirclement'),
        'config',
        'filters.yaml')

    with open(filter_yaml, 'r') as ymlfile:
        filter_yaml_content = yaml.safe_load(ymlfile)

    robots_list = [robot for robot in crazyflies['robots'] if crazyflies['robots'][robot]['enabled']]
    for robot in robots_list:
        Nodes.append(Node(
            package='crazy_encirclement',
            executable='circle_distortion',
            name=robot+'_circle_distortion',
            output='screen',
            parameters=[{'robot': robot, 'number_of_agents': len(robots_list)} + filter_yaml_content.get('circle_distortion', {}).get('ros__parameters', {})],
            ))
        Nodes.append(Node(
            package='crazy_encirclement',
            executable='gps.py',
            name=robot+'_gps_node',
            output='screen',
            parameters=[{'robot': robot} + filter_yaml_content.get('circle_distortion', {}).get('ros__parameters', {})],
        ))
        Nodes.append(Node(
            package='crazyflie',
            executable='watch_dog.py',
            name=robot+'_watch_dog',
            output='screen',
            parameters=[{'robot_prefix': robot}]
        ))
    
    Nodes.append(Node(
        package='crazy_encirclement',
        executable='agents_order',
        name='agents_order',
        output='screen',
        parameters= [{'robot_data': robots_list}]
    ))

    return Nodes


def generate_launch_description():
    default_crazyflies_yaml_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'crazyflies.yaml')
    
    default_motion_capture_yaml_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'motion_capture.yaml')

    default_rviz_config_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'config.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('crazyflies_yaml_file', default_value=default_crazyflies_yaml_path),
        DeclareLaunchArgument('motion_capture_yaml_file', default_value=default_motion_capture_yaml_path),
        DeclareLaunchArgument('rviz_config_file', default_value=default_rviz_config_path),
        DeclareLaunchArgument('backend', default_value='cpp'),
        DeclareLaunchArgument('debug', default_value='False'),
        DeclareLaunchArgument('rviz', default_value='False'),
        DeclareLaunchArgument('mocap', default_value='True'),
        OpaqueFunction(function=parse_yaml),
        Node(
            condition=LaunchConfigurationEquals('rviz', 'True'),
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config_file')],
            parameters=[{
                "use_sim_time": PythonExpression(["'", LaunchConfiguration('backend'), "' == 'sim'"]),
            }]
        ),
    ])

# Example commands to run the launch file or node with parameters and remapping:
#   ros2 launch crazy_encirclement <launch_file>.launch.py --ros-args -p robot:=C05 --remap /old_topic:=/new_topic
#   ros2 run crazy_encirclement encirclement --ros-args -p robot:='C05' --remap /encirclement:=/C05/encirclement
