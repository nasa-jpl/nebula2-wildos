from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    gps_save_path = LaunchConfiguration('gps_save_path')
    metrics_path = LaunchConfiguration('metrics_path')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'save_path',
            default_value='gps_path.json',
            description='Path to save GPS coordinates'
        ),

        Node(
            package='img_vlms',
            executable='gps_save',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'save_path': gps_save_path}
            ],
        ),

        Node(
            package='gps_visualization',
            executable='gps_path_pub',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
        ),

        Node(
            package='img_vlms',
            executable='metrics_save',
            output='screen',
            parameters=[
                {'save_path': metrics_path},
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
