from launch import LaunchDescription
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    ns = LaunchConfiguration('ns')
    do_object_search = LaunchConfiguration('do_object_search')
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'ns',
            default_value='',
            description='Robot namespace'
        ),
        DeclareLaunchArgument(
            'do_object_search',
            default_value='false',
            description='Enable object search and launch obj_mask_triangulation node'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='Logging level (DEBUG, INFO, WARN, ERROR, FATAL)'
        ),

        # Always launch wildos
        Node(
            package='visual_navigation',
            executable='wildos',
            output='screen',
            arguments=[
                '--config', 'wildos_nav_conf.yaml',
                '--do_object_search', do_object_search,
                '--ros-args', '--log-level', log_level
            ],
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/tf', PathJoinSubstitution([TextSubstitution(text='/'), ns, TextSubstitution(text='tf')])),
                ('/tf_static', PathJoinSubstitution([TextSubstitution(text='/'), ns, TextSubstitution(text='tf_static')])),
            ]
        ),

        # Conditionally launch obj_mask_triangulation
        Node(
            package='visual_navigation',
            executable='obj_mask_triangulation',
            output='screen',
            arguments=[
                '--ros-args', '--log-level', log_level
            ],
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/tf', PathJoinSubstitution([TextSubstitution(text='/'), ns, TextSubstitution(text='tf')])),
                ('/tf_static', PathJoinSubstitution([TextSubstitution(text='/'), ns, TextSubstitution(text='tf_static')])),
            ],
            condition=IfCondition(do_object_search)
        ),
    ])
