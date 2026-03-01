from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_namespace = LaunchConfiguration('robot_namespace')
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='',
            description='Robot namespace'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='Logging level (DEBUG, INFO, WARN, ERROR, FATAL)'
        ),

        # Always launch lrn
        Node(
            package='visual_navigation',
            executable='radio_triangulate',
            name='radio_triangulator',
            output='screen',
            arguments=[
                '--config', 'radio_triangulator_conf.yaml',
                '--ros-args', '--log-level', log_level
            ],
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_name': robot_namespace},
            ],
            namespace=robot_namespace,
            remappings=[
                ('/tf', PathJoinSubstitution([TextSubstitution(text='/'), robot_namespace, TextSubstitution(text='tf')])),
                ('/tf_static', PathJoinSubstitution([TextSubstitution(text='/'), robot_namespace, TextSubstitution(text='tf_static')])),
                ('/diagnostics', PathJoinSubstitution([TextSubstitution(text='/'), robot_namespace, TextSubstitution(text='diagnostics')])),
                ('/parameter_events', PathJoinSubstitution([TextSubstitution(text='/'), robot_namespace, TextSubstitution(text='parameter_events')])),
                ('/rosout', PathJoinSubstitution([TextSubstitution(text='/'), robot_namespace, TextSubstitution(text='rosout')])),
            ]
        ),
    ])
