import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument('name', default_value='quadrotor'),
        DeclareLaunchArgument('platform_type', default_value='mujoco'),
        DeclareLaunchArgument('world_frame_id', default_value='world'),
        DeclareLaunchArgument('rate_odom', default_value='200.0'),
        DeclareLaunchArgument('rate_imu', default_value='500.0'),
        DeclareLaunchArgument('flag_build', default_value='True'),
        DeclareLaunchArgument('takeoff_height', default_value='1.0'),
    ]

    name = LaunchConfiguration('name')
    platform_type = LaunchConfiguration('platform_type')
    world_frame_id = LaunchConfiguration('world_frame_id')
    rate_odom = LaunchConfiguration('rate_odom')
    rate_imu = LaunchConfiguration('rate_imu')
    flag_build = LaunchConfiguration('flag_build')
    takeoff_height = LaunchConfiguration('takeoff_height')

    source_model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'model',
        'mujoco',
        'drone_ground.xml',
    )
    installed_model_path = PathJoinSubstitution([
        get_package_share_directory('dq_nmpc'),
        'model',
        platform_type,
        'drone_ground.xml',
    ])
    model_path = source_model_path if os.path.exists(source_model_path) else installed_model_path
    package_share_dir = get_package_share_directory('dq_nmpc')
    source_run_dir = os.path.abspath(
        os.path.join(package_share_dir, '..', '..', '..', 'src', 'dq_nmpc')
    )
    run_dir = source_run_dir if os.path.exists(source_run_dir) else package_share_dir

    launch_args.append(
        DeclareLaunchArgument('model_path', default_value=model_path)
    )
    launch_args.append(
        DeclareLaunchArgument('run_dir', default_value=run_dir)
    )

    control_config = PathJoinSubstitution([
        get_package_share_directory('dq_nmpc'),
        TextSubstitution(text='config'),
        platform_type,
        TextSubstitution(text='default'),
        TextSubstitution(text='dq_control.yaml'),
    ])

    trackers_config = PathJoinSubstitution([
        get_package_share_directory('trackers_manager'),
        'config',
        'trackers.yaml',
    ])

    tracker_param_config = PathJoinSubstitution([
        get_package_share_directory('trackers_manager'),
        'config',
        'tracker_param.yaml',
    ])

    mav_manager_config = PathJoinSubstitution([
        get_package_share_directory('mav_manager'),
        'config',
        'mav_manager_params.yaml',
    ])

    quadrotor_simulator_mujoco_node = Node(
        package='quadrotor_simulator_mujoco',
        executable='quadrotor_simulator',
        name='quadrrotor_simulator',
        namespace=name,
        output='screen',
        arguments=[LaunchConfiguration('model_path')],
        parameters=[{
            'rate_odom': rate_odom,
            'rate_imu': rate_imu,
            'world_frame_id': world_frame_id,
            'body_frame_id': name,
            'use_internal_so3_control': False,
        }],
    )

    nmpc_controller_node = Node(
        package='dq_nmpc',
        executable='dq_nmpc',
        name='dq_controller',
        namespace=name,
        output='screen',
        cwd=LaunchConfiguration('run_dir'),
        parameters=[
            control_config,
            {
                'world_frame_id': world_frame_id,
                'body_frame_id': name,
                'flag_build': flag_build,
                'wait_for_reference': True,
                'reference_timeout_sec': 0.25,
            },
        ],
    )

    trackers_manager_component = ComposableNode(
        package='trackers_manager',
        plugin='trackers_manager::TrackersManager',
        namespace=name,
        name='trackers_manager',
        remappings=[('cmd', 'position_cmd')],
        parameters=[trackers_config, tracker_param_config, control_config],
    )

    trackers_manager_container = ComposableNodeContainer(
        name='trackers_manager_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[trackers_manager_component],
        output='screen',
    )

    mav_manager_component = ComposableNode(
        package='mav_manager',
        plugin='mav_manager::MAVManager',
        namespace=name,
        name='mav_manager',
        parameters=[
            mav_manager_config,
            control_config,
            {
                'takeoff_height': takeoff_height,
            },
        ],
    )

    mav_manager_container = ComposableNodeContainer(
        name='mav_manager_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[mav_manager_component],
        output='screen',
    )

    debug_print = LogInfo(
        msg=['[INFO] Using model path: ', LaunchConfiguration('model_path')]
    )

    ld = LaunchDescription(launch_args)
    ld.add_action(debug_print)
    ld.add_action(quadrotor_simulator_mujoco_node)
    ld.add_action(trackers_manager_container)
    ld.add_action(TimerAction(period=1.0, actions=[mav_manager_container]))
    ld.add_action(nmpc_controller_node)
    return ld
