
import os
import re
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
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
    start_crazysim = LaunchConfiguration('start_crazysim').perform(
        context).lower() in ('true', '1', 'yes')
    limo_pose_port = int(
        LaunchConfiguration('crazysim_limo_pose_port').perform(context))

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
        # CrazySim provides state through the firmware estimator.  The CFLib
        # server only emits each drone's PoseStamped and world->drone TF when
        # the firmware pose log is enabled for that robot type.
        for robot_name, robot_config in crazyflies['robots'].items():
            if not robot_config.get('enabled', False):
                continue
            robot_type = crazyflies['robot_types'][robot_config['type']]
            firmware_logging = robot_type.setdefault('firmware_logging', {})
            firmware_logging['enabled'] = True
            default_topics = firmware_logging.setdefault(
                'default_topics', {})
            default_topics['pose'] = {'frequency': 10}

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

    enabled_robots = [
        name for name, config in crazyflies['robots'].items()
        if config['enabled']
    ]

    if use_sim_enabled and start_crazysim:
        firmware_path = os.path.abspath(os.path.expanduser(
            LaunchConfiguration('crazysim_firmware_path').perform(context)))
        coordinates_path = os.path.abspath(os.path.expanduser(
            LaunchConfiguration('crazysim_coordinates_file').perform(context)))
        model = LaunchConfiguration('crazysim_model').perform(context)
        limo_model_path = os.path.abspath(os.path.expanduser(
            LaunchConfiguration('crazysim_limo_model_path').perform(context)))
        launcher = os.path.join(
            firmware_path,
            'tools',
            'crazyflie-simulation',
            'simulator_files',
            'mujoco',
            'launch',
            'sitl_multiagent_text.sh')
        firmware_binary = os.path.join(
            firmware_path, 'sitl_make', 'build', 'cf2')

        if not os.path.isfile(launcher):
            raise FileNotFoundError(
                f'CrazySim MuJoCo launcher not found: {launcher}')
        if not os.path.isfile(firmware_binary):
            raise FileNotFoundError(
                'CrazySim firmware is not built. Expected executable at '
                f'{firmware_binary}')
        if not os.path.isfile(limo_model_path):
            raise FileNotFoundError(
                f'LIMO MuJoCo model not found: {limo_model_path}')

        coordinates_directory = os.path.dirname(coordinates_path)
        if coordinates_directory:
            os.makedirs(coordinates_directory, exist_ok=True)

        with open(coordinates_path, 'w', encoding='utf-8') as coordinates_file:
            for index, robot in enumerate(enabled_robots):
                uri = str(crazyflies['robots'][robot]['uri'])
                match = re.fullmatch(r'udp://[^:]+:(\d+)', uri)
                expected_port = 19850 + index
                if match is None or int(match.group(1)) != expected_port:
                    raise ValueError(
                        f'{robot} must use UDP port {expected_port} to match '
                        f'CrazySim spawn index {index}; configured URI is {uri}.')

                position = crazyflies['robots'][robot]['initial_position']
                coordinates_file.write(
                    f'{float(position[0]):.9f},{float(position[1]):.9f}\n')
                print(
                    f'CrazySim agent {index}: {robot}, port={expected_port}, '
                    f'position=({float(position[0]):.3f}, '
                    f'{float(position[1]):.3f})')

        Nodes.append(ExecuteProcess(
            cmd=[
                'bash', launcher,
                '-m', model,
                '-f', coordinates_path,
                '--limo-model', limo_model_path,
                '--limo-pose-port', str(limo_pose_port),
            ],
            cwd=firmware_path,
            output='screen',
            additional_env={
                'PATH': os.pathsep.join(filter(None, [
                    LaunchConfiguration(
                        'crazysim_python_bin').perform(context),
                    os.environ.get('PATH', ''),
                ])),
            },
        ))

    Nodes.append(Node(
            package='motion_capture_tracking',
            executable='motion_capture_tracking_node',
            condition=IfCondition(PythonExpression(["'", LaunchConfiguration('backend'), "' != 'sim' and '", LaunchConfiguration('mocap'), "' == 'True'"])),
            name='motion_capture_tracking',
            output='screen',
            parameters= [motion_capture_params],
        ))
    cflib_server = Node(
            package='crazyflie_server_py',
            executable='crazyflie_server',
            condition=LaunchConfigurationEquals('backend','cflib'),
            name='crazyflie_server',
            output='screen',
            parameters= server_params,
            additional_env={
                'PYTHONPATH': os.pathsep.join(filter(None, [
                    LaunchConfiguration('cflib_pythonpath').perform(context),
                    os.environ.get('PYTHONPATH', ''),
                ])),
            },
        )
    if use_sim_enabled and start_crazysim:
        Nodes.append(TimerAction(
            period=float(LaunchConfiguration(
                'crazysim_server_delay').perform(context)),
            actions=[cflib_server],
        ))
    else:
        Nodes.append(cflib_server)
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

    Nodes.append(Node(
            package='target_protection',
            executable='sim_point_mass',
            condition=IfCondition(LaunchConfiguration('use_sim')),
            name='sim_point_mass',
            output='screen',
            parameters=[{
                'name': sim_target_name,
                'randomize_initial_position': False,
                'initial_x': float(sim_target_position[0]),
                'initial_y': float(sim_target_position[1]),
                'circle_radius': 2.0,
                'angular_velocity': 0.2,
                'mujoco_pose_enabled': start_crazysim,
                'mujoco_pose_port': limo_pose_port,
                'use_sim_time': False,
            }],
        ))
    Nodes.append(Node(
            package='target_protection',
            executable='sim_pose_bridge',
            condition=IfCondition(LaunchConfiguration('use_sim')),
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
        DeclareLaunchArgument('start_crazysim', default_value='True'),
        DeclareLaunchArgument(
            'crazysim_firmware_path',
            default_value=os.path.expanduser(
                '~/ros2_ws/src/CrazySim/crazyflie-firmware')),
        DeclareLaunchArgument(
            'crazysim_coordinates_file',
            default_value='/tmp/target_protection_crazysim_layout.txt'),
        DeclareLaunchArgument(
            'cflib_pythonpath',
            default_value=os.path.expanduser(
                '~/venvs/crazyflie/lib/python3.12/site-packages')),
        DeclareLaunchArgument(
            'crazysim_python_bin',
            default_value=os.path.expanduser('~/venvs/crazyflie/bin')),
        DeclareLaunchArgument('crazysim_model', default_value='cf21B_500'),
        DeclareLaunchArgument(
            'crazysim_limo_model_path',
            default_value=os.path.expanduser(
                '~/ros2_ws/src/limo_ros2/limo_description/mujoco/limo.xml')),
        DeclareLaunchArgument(
            'crazysim_limo_pose_port', default_value='19849'),
        DeclareLaunchArgument('crazysim_server_delay', default_value='5.0'),
        DeclareLaunchArgument(
            'simulation_controller_delay', default_value='10.0'),
        DeclareLaunchArgument(
            'simulation_experiment_file',
            default_value=default_simulation_experiment_path),
        SetLaunchConfiguration(
            'backend', 'cflib', condition=IfCondition(LaunchConfiguration('use_sim'))),
        SetLaunchConfiguration(
            'mocap', 'False', condition=IfCondition(LaunchConfiguration('use_sim'))),
        SetLaunchConfiguration(
            'rviz', 'True', condition=IfCondition(LaunchConfiguration('use_sim'))),
        OpaqueFunction(function=parse_yaml),
        TimerAction(
            period=LaunchConfiguration('simulation_controller_delay'),
            actions=[Node(
                condition=IfCondition(LaunchConfiguration('use_sim')),
                package='target_protection',
                executable='simulation_experiment_controller',
                name='simulation_experiment_controller',
                output='screen',
                parameters=[LaunchConfiguration('simulation_experiment_file')]
            )],
        ),
        Node(
            condition=LaunchConfigurationEquals('rviz', 'True'),
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config_file')],
            parameters=[{
                "use_sim_time": False,
            }]
        ),
    ])
