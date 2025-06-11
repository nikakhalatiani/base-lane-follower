#!/usr/bin/env python3
"""
Wheel control node for Duckietown robot.
Receives motor commands and publishes them to the wheel driver.
"""

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from config import ROSConfig

class WheelControlNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(WheelControlNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        
        # Setup ROS components
        self._setup_ros_components()
        
        # Initialize motor velocities
        self._vel_left = 0
        self._vel_right = 0
        
        # Initialize CV bridge (legacy - could be removed if not needed)
        self.bridge = CvBridge()

    def _setup_ros_components(self):
        # Get vehicle name from environment
        self.vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"

        # Setup publisher for wheel commands
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=ROSConfig.PUBLISHER_QUEUE_SIZE)

        # Setup subscribers for motor commands
        self.left_motor = rospy.Subscriber(
            "left_motor", Float64, self.callback_left)
        self.right_motor = rospy.Subscriber(
            "right_motor", Float64, self.callback_right)

    def callback_left(self, msg):
        self._vel_left = msg.data

    def callback_right(self, msg):
        self._vel_right = msg.data

    def run(self):
        # Publish messages at control frequency
        rate = rospy.Rate(ROSConfig.CONTROL_FREQUENCY)
        
        while not rospy.is_shutdown():
            # Create and publish wheel command message
            message = WheelsCmdStamped(
                vel_left=self._vel_left, 
                vel_right=self._vel_right
            )
            self._publisher.publish(message)
            rate.sleep()

    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)


if __name__ == '__main__':
    # Create and run the node
    node = WheelControlNode(node_name='wheel_control_node')
    
    # Register shutdown hook
    rospy.on_shutdown(node.on_shutdown)
    
    # Start the control loop
    node.run()
    
    # Keep the process from terminating
    rospy.spin()