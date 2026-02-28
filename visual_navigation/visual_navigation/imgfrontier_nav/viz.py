import numpy as np

from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped

def get_path_msg(path: np.ndarray, frame_id, stamp):
    path_msg = PathMsg()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = stamp
    path_msg.poses = []
    for p in path:
        pose = PoseStamped()
        pose.pose.position.x = p[0]
        pose.pose.position.y = p[1]
        pose.pose.position.z = p[2]
        path_msg.poses.append(pose)
    return path_msg