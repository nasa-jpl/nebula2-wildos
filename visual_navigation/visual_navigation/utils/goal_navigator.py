
from omegaconf import OmegaConf
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry

from img_vlms.utils.tf_lookup_sub import TFLookupSubscriber

# TODO: waypoints are assumed in the global frame, change this if needed

class GoalNavigator(TFLookupSubscriber):
    default_goalnav_config = {
        # Nav
        "goal_reach_radius": 2.0,
        "reach_in_2D": True,
        "waypoints": None,
        "overwrite_waypoints": True,

        # ROS2
        "waypoint_frame": "spot1/odom",
        "waypoint_topic": "waypoint",
        "waypoint_viz_topic": "waypoint_viz",
        "goal_dir_viz_topic": "goal_direction"
    }
    def __init__(self, node_name:str, nav_config: OmegaConf, tf_lookup_config: OmegaConf):
        super().__init__(node_name=node_name, config=tf_lookup_config)
        config = OmegaConf.merge(OmegaConf.create(self.default_goalnav_config), nav_config)

        self.goal_reach_radius = config.goal_reach_radius
        self.global_frame = config.waypoint_frame
        self.reach_in_2D = config.reach_in_2D
        self.overwrite_waypoints = config.overwrite_waypoints

        if config.waypoints is None:
            self.waypoints = None
            self.current_wp_idx = None
        else:
            self.waypoints = np.array(config.waypoints)
            self.current_wp_idx = 0

        self.waypoint_sub = self.create_subscription(
            PoseStamped,
            config.waypoint_topic,
            self.waypoint_callback,
            10
        )
        self.waypoint_viz_pub = self.create_publisher(
            MarkerArray,
            config.waypoint_viz_topic,
            10
        )
        self.goal_direction_pub = self.create_publisher(
            Marker,
            config.goal_dir_viz_topic,
            10
        )

    def waypoint_callback(self, msg):
        waypoint = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        if self.waypoints is None or self.overwrite_waypoints:
            self.waypoints = waypoint.reshape(1, 3)
            self.current_wp_idx = 0
        else:
            self.waypoints = np.vstack((self.waypoints, waypoint))
        self.get_logger().info(f"Received new waypoint: {waypoint}, total waypoints: {len(self.waypoints)}")
        self.publish_goal_waypoints()

    def publish_goal_waypoints(self):
        if self.waypoints is None:
            return

        marker_array = MarkerArray()
        for i, waypoint in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "goal_waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = waypoint[2]
            marker.scale.x = 5.0
            marker.scale.y = 5.0
            marker.scale.z = 5.0
            marker.color.a = 1.0
            marker.color.r = 0.0 if i == self.current_wp_idx else 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
        self.waypoint_viz_pub.publish(marker_array)

    def publish_goal_direction(self, current_pos, goal_pos):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point()
        start.x, start.y, start.z = current_pos
        end = Point()
        end.x, end.y, end.z = goal_pos
        marker.points.append(start)
        marker.points.append(end)
        marker.scale.x = 1.0
        marker.scale.y = 1.5
        marker.scale.z = 1.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.goal_direction_pub.publish(marker)

    def compute_goal_range_heading(self, current_odom_msg: Odometry):
        """
        Compute the range and heading to the current goal waypoint.
        We assume that the current_odom_msg is in the same frame as the waypoints.
        TODO: remove assumption and transform goal to odom frame using TF.
        """
        assert current_odom_msg.header.frame_id == self.global_frame, \
            f"Expected frame_id {self.global_frame}, but got {current_odom_msg.header.frame_id}"

        current_pos = np.array([
            current_odom_msg.pose.pose.position.x,
            current_odom_msg.pose.pose.position.y,
            current_odom_msg.pose.pose.position.z
        ])
        current_waypoint = self.waypoints[self.current_wp_idx]
        self.publish_goal_direction(current_pos, current_waypoint)

        goal_heading = current_waypoint - current_pos
        if self.reach_in_2D:
            goal_heading[2] = 0.0
        goal_error = np.linalg.norm(goal_heading)
        goal_heading = goal_heading / (goal_error + 1e-6)

        if goal_error < self.goal_reach_radius:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.get_logger().warn("Reached the last waypoint.")
                self.current_wp_idx = None
                self.waypoints = None
            else:
                self.get_logger().info(f"Moving to waypoint {self.current_wp_idx}")

        return goal_error, goal_heading