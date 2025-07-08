"""
Save a path published in ROS
"""
#!/usr/bin/env python
import rospy
from nav_msgs.msg import Path
import csv
import os

PATH_TOPIC = '/genz/trajectory'


class PathLogger:
    def __init__(self):
        # Parámetros
        self.output_file = os.path.expanduser("path_log.csv")
        self.subscriber = rospy.Subscriber(PATH_TOPIC, Path, self.callback)

        # Crear o limpiar el fichero CSV
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['#timestamp [ns]', 'x', 'y', 'z',
                             'qx', 'qy', 'qz', 'qw'])

    def callback(self, msg):
        print('Received trajectory!')
        with open(self.output_file, 'w') as f:
            print('Saving trajectory!')
            print('Received traj. length: ', len(msg.poses))
            writer = csv.writer(f)
            for pose_stamped in msg.poses:
                t = int(pose_stamped.header.stamp.to_sec()*1e9)
                pos = pose_stamped.pose.position
                ori = pose_stamped.pose.orientation
                writer.writerow([t, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        rospy.loginfo("Path written with {} poses.".format(len(msg.poses)))


if __name__ == '__main__':
    rospy.init_node('path_logger')
    PathLogger()
    rospy.spin()
