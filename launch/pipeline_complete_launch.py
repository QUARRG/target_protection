
import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import LaunchConfigurationEquals
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def parse_yaml(context):
    use_sim_enabled = LaunchConfiguration('use_sim').perform(
        context).lower() in ('true', '1', 'yes')
    use_sim = ParameterValue(
        LaunchConfiguration('use_sim'),
        value_type=bool)

    # Load the crazyflies YAML file
    crazyflies_yaml = LaunchConfiguration('crazyflies_yaml_file').perform(context)
    with open(crazyflies_yaml, 'r') as file:
        crazyflies = yaml.safe_load(file)

    # Load filter settings before constructing the simulation layout.
    filter_yaml = os.path.join(
        get_package_share_directory('target_protection'),
        'config',
        'filters.yaml')
    with open(filter_yaml, 'r') as ymlfile:
        filter_yaml_content = yaml.safe_load(ymlfile)
    experiment_settings = filter_yaml_content.get(
        'FollowLimoEncirclementFilterUnicycle', {})

    robots_list = []
    evader = None
    for robot, configuration in crazyflies['robots'].items():
        if not configuration['enabled']:
            continue
        if configuration.get('role') == 'pursuer':
            robots_list.append(robot)
        elif configuration.get('role') == 'evader':
            evader = robot

    sim_target_name = 'LIMO'
    sim_target_position = [0.0, 0.0]
    if use_sim_enabled:
        layout_seed = int(
            LaunchConfiguration('simulation_layout_seed').perform(context))
        generator = np.random.default_rng(
            None if layout_seed < 0 else layout_seed)

        while True:
            sim_target_position = generator.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(sim_target_position) >= 0.1:
                break

        radius_nominal = float(
            experiment_settings.get('controls', {}).get(
                'radius_nominal', 0.5))

        evader_distance = float(
            LaunchConfiguration('evader_initial_distance').perform(context))
        if evader_distance < 4.0:
            raise ValueError('evader_initial_distance must be at least 4.0 m.')
        if evader is not None:
            evader_phase_degrees = int(generator.integers(0, 360))
            evader_phase = np.deg2rad(evader_phase_degrees)
            evader_position = crazyflies['robots'][evader]['initial_position']
            evader_position[0] = float(
                sim_target_position[0]
                + evader_distance * np.cos(evader_phase))
            evader_position[1] = float(
                sim_target_position[1]
                + evader_distance * np.sin(evader_phase))
            print(
                f'{evader} evader initial layout: distance={evader_distance:.3f} m, '
                f'phase={evader_phase_degrees} deg, position={evader_position}')

        for robot in robots_list:
            radius = generator.uniform(
                0.5 * radius_nominal, 2.0 * radius_nominal)
            phase_degrees = int(generator.integers(0, 360))
            phase = np.deg2rad(phase_degrees)
            initial_position = crazyflies['robots'][robot]['initial_position']
            initial_position[0] = float(
                sim_target_position[0] + radius * np.cos(phase))
            initial_position[1] = float(
                sim_target_position[1] + radius * np.sin(phase))
            print(
                f'{robot} initial layout: radius={radius:.3f} m, '
                f'phase={phase_degrees} deg, position={initial_position}')

        print(
            f'{sim_target_name} initial layout: '
            f'position=({sim_target_position[0]:.3f}, '
            f'{sim_target_position[1]:.3f}, 0.000)')

    # server params
    server_yaml = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'server.yaml')

    with open(server_yaml, 'r') as ymlfile:
        server_yaml_content = yaml.safe_load(ymlfile)

    if use_sim_enabled:
        server_yaml_content[
            '/crazyflie_server']['ros__parameters']['sim']['controller'] = 'pid'

    server_params = [crazyflies] + [server_yaml_content['/crazyflie_server']['ros__parameters']]
    # robot description
    urdf = os.path.join(
        get_package_share_directory('crazyflie_description'),
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

    enabled_robots = [
        name for name, config in crazyflies['robots'].items()
        if config['enabled']
    ]
    Nodes.append(Node(
            package='target_protection',
            executable='sim_point_mass',
            condition=LaunchConfigurationEquals('backend', 'sim'),
            name='sim_point_mass',
            output='screen',
            parameters=[{
                'name': sim_target_name,
                'randomize_initial_position': False,
                'initial_x': float(sim_target_position[0]),
                'initial_y': float(sim_target_position[1]),
                'circle_radius': 2.0,
                'angular_velocity': 0.5,
                'use_sim_time': True,
            }],
        ))
    Nodes.append(Node(
            package='target_protection',
            executable='sim_pose_bridge',
            condition=LaunchConfigurationEquals('backend', 'sim'),
            name='sim_pose_bridge',
            output='screen',
            parameters=[{'robot_names': enabled_robots + [sim_target_name]}],
        ))

    print(f'Robots in the encirclement: {robots_list}, Evader: {evader}')

    for robot in robots_list:
        # Nodes for each robot
        Nodes.append(Node(
            package='target_protection',
            executable='pipeline_complete',
            name=robot+'_pipeline_complete',
            output='screen',
            parameters=[{'robot': robot,
                         'number_of_agents': len(robots_list),
                         'target': evader,
                         'use_sim': use_sim} | filter_yaml_content.get('FollowLimoEncirclementFilterUnicycle', {})]
        ))

        # GPS/Scanner II Node for each robot
        scanner_update_hz = filter_yaml_content.get('FollowLimoEncirclementFilterUnicycle', {}).get('others', {}).get('update_hz', 10.0)
        Nodes.append(Node(
            package='target_protection',
            executable='gps_scanner_ii',
            name=robot+'_gps_scanner_ii_node',
            output='screen',
            parameters=[{'robot': robot, 'update_hz': scanner_update_hz}],
        ))

        # Watch dog node for each robot
        Nodes.append(Node(
            package='controller_pkg',
            executable='watch_dog',
            name=robot+'_watch_dog',
            output='screen',
            parameters=[{'robot_prefix': robot}]
        ))

    if evader is not None:
        # Watch dog node for evader robot
        Nodes.append(Node(
            package='controller_pkg',
            executable='watch_dog',
            name=evader+'_watch_dog',
            output='screen',
            parameters=[{'robot_prefix': evader}]
            ))
        Nodes.append(Node(
            package='target_protection',
            executable='evader',
            name=evader+'_evader',
            output='screen',
            parameters=[{'evader': evader,
                         'use_sim': use_sim} | filter_yaml_content.get('FollowLimoEncirclementFilterUnicycle', {})]
        ))
        
    # Agents order node
    Nodes.append(Node(
        package='target_protection',
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

    default_simulation_experiment_path = os.path.join(
        get_package_share_directory('target_protection'),
        'config',
        'simulation_experiment.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('crazyflies_yaml_file', default_value=default_crazyflies_yaml_path),
        DeclareLaunchArgument('motion_capture_yaml_file', default_value=default_motion_capture_yaml_path),
        DeclareLaunchArgument('rviz_config_file', default_value=default_rviz_config_path),
        DeclareLaunchArgument('backend', default_value='cpp'),
        DeclareLaunchArgument('debug', default_value='False'),
        DeclareLaunchArgument('rviz', default_value='False'),
        DeclareLaunchArgument('mocap', default_value='True'),
        DeclareLaunchArgument('use_sim', default_value='False'),
        DeclareLaunchArgument('simulation_layout_seed', default_value='-1'),
        DeclareLaunchArgument('evader_initial_distance', default_value='10.0'),
        DeclareLaunchArgument(
            'simulation_experiment_file',
            default_value=default_simulation_experiment_path),
        SetLaunchConfiguration(
            'backend', 'sim', condition=IfCondition(LaunchConfiguration('use_sim'))),
        SetLaunchConfiguration(
            'mocap', 'False', condition=IfCondition(LaunchConfiguration('use_sim'))),
        SetLaunchConfiguration(
            'rviz', 'True', condition=IfCondition(LaunchConfiguration('use_sim'))),
        OpaqueFunction(function=parse_yaml),
        Node(
            condition=IfCondition(LaunchConfiguration('use_sim')),
            package='target_protection',
            executable='simulation_experiment_controller',
            name='simulation_experiment_controller',
            output='screen',
            parameters=[LaunchConfiguration('simulation_experiment_file')]
        ),
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
