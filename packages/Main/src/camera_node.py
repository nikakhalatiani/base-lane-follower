#!/usr/bin/env python3
import os
import time
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from collections import deque
from image_processor import ImageProcessor
from lane_detector import LaneDetector
from pid_controller import PDController
from config import AprilTagConfig

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        # Initialize processors
        self.image_processor = ImageProcessor()
        self.lane_detector = LaneDetector()
        self.pid_controller = PDController()
        
        # Track previous error for lane detection
        self.prev_error = 0
        
        self.stop_sign_detected = False
        self.stop_start_time = None
        self.is_stopping = False
        self.last_stop_time = None
        
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._window = "camera-reader"
        self.bridge = CvBridge()
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        self.sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.callback)
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

        self.left_motor = rospy.Publisher("left_motor", Float64, queue_size=1)
        self.right_motor = rospy.Publisher("right_motor", Float64, queue_size=1)

        self.shutting_down = False
        self.vehicle_running = False  # Start/stop state
        self.color_picker_active = False
        
        # Create compact trackbars for color adjustment
        # Yellow HSV - Compact layout
        cv2.createTrackbar('Y_H', self._window, 22, 40, lambda x: None)
        cv2.createTrackbar('Y_S', self._window, 72, 255, lambda x: None)
        cv2.createTrackbar('Y_V', self._window, 119, 255, lambda x: None)
        cv2.createTrackbar('Y_R', self._window, 5, 20, lambda x: None)
        
        # White HLS - Compact layout  
        cv2.createTrackbar('W_H', self._window, 47, 180, lambda x: None)
        cv2.createTrackbar('W_L', self._window, 115, 255, lambda x: None)
        cv2.createTrackbar('W_S', self._window, 128, 255, lambda x: None)
        cv2.createTrackbar('W_R', self._window, 123, 180, lambda x: None)
        
        # Control sliders
        cv2.createTrackbar('Speed', self._window, 18, 50, lambda x: None)  # Speed in cm/s * 100 (max 0.50)
        cv2.createTrackbar('Start', self._window, 0, 1, self.toggle_start_stop)

        rospy.on_shutdown(self.shutdown_hook)

    def toggle_start_stop(self, value):
        """Toggle vehicle start/stop state"""
        self.vehicle_running = bool(value)
        if not self.vehicle_running:
            # Stop immediately when toggled off
            self.left_motor.publish(0)
            self.right_motor.publish(0)
    
    def shutdown_hook(self):
        self.shutting_down = True
        self.left_motor.publish(0)
        self.right_motor.publish(0)
        cv2.destroyAllWindows()

    def callback(self, msg):
        if self.shutting_down:
            return

        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        vis_image = self.image.copy()
        
        # Red light detection
        red_light_detected = self.image_processor.detect_red_light(self.image)
        
        # AprilTag detection and stop sign handling
        try:
            apriltag_detections = self.image_processor.detect_apriltags(self.image)
            self.image_processor.add_apriltag_visualization(vis_image, apriltag_detections)
            
            stop_sign_found, tag_size = self.image_processor.check_stop_sign(apriltag_detections)
            
            current_time = time.time()
            can_stop = True
            
            if self.last_stop_time is not None:
                time_since_last_stop = current_time - self.last_stop_time
                can_stop = time_since_last_stop >= AprilTagConfig.STOP_COOLDOWN
            
            if stop_sign_found and not self.is_stopping and can_stop:
                print(f"Stop sign detected! Tag ID: {AprilTagConfig.STOP_SIGN_ID}, Size: {tag_size:.1f}")
                self.stop_sign_detected = True
                self.stop_start_time = current_time
                self.is_stopping = True
            elif stop_sign_found and not can_stop:
                time_remaining = AprilTagConfig.STOP_COOLDOWN - (current_time - self.last_stop_time)
                print(f"Stop sign visible but in cooldown. {time_remaining:.1f}s remaining")
            
            if self.is_stopping:
                elapsed_time = current_time - self.stop_start_time
                if elapsed_time >= AprilTagConfig.STOP_DURATION:
                    print(f"Stop duration complete ({AprilTagConfig.STOP_DURATION}s). Resuming movement.")
                    self.is_stopping = False
                    self.stop_sign_detected = False
                    self.last_stop_time = current_time
                    self.stop_start_time = None
                else:
                    print(f"Stopping... {elapsed_time:.1f}/{AprilTagConfig.STOP_DURATION}s")
                    
        except Exception as e:
            print(f"AprilTag processing error: {e}")
            apriltag_detections = []
        
        # Read compact slider values
        y_h = cv2.getTrackbarPos('Y_H', self._window)
        y_s = cv2.getTrackbarPos('Y_S', self._window)
        y_v = cv2.getTrackbarPos('Y_V', self._window)
        y_range = cv2.getTrackbarPos('Y_R', self._window)
        
        w_h = cv2.getTrackbarPos('W_H', self._window)
        w_l = cv2.getTrackbarPos('W_L', self._window)
        w_s = cv2.getTrackbarPos('W_S', self._window)
        w_range = cv2.getTrackbarPos('W_R', self._window)
        
        # Read control values
        speed_setting = cv2.getTrackbarPos('Speed', self._window) / 100.0  # Convert back to 0.00-0.30
        
        slider_values = {
            'yellow_hsv': [[max(0, y_h - y_range), y_s, y_v], [min(179, y_h + y_range), 255, 255]],
            'white_hls': [[max(0, w_h - w_range), w_l, 0], [min(179, w_h + w_range), 255, w_s]]
        }
        
        # Use the new lane detection pipeline with slider values
        detection_results = self.lane_detector.process_frame(self.image, self.prev_error, slider_values)
        self.prev_error = detection_results['error']
        
        # Use the new PID controller with speed setting
        left_motor, right_motor, steering, debug_info = self.pid_controller.process_control_loop(detection_results, speed_setting)
        
        # Override for special conditions
        if not self.vehicle_running:
            left_motor = 0.0
            right_motor = 0.0
        elif self.is_stopping:
            left_motor = 0.0
            right_motor = 0.0
        elif red_light_detected:
            left_motor = 0.0
            right_motor = 0.0
        
        # Publish motor commands
        if not self.shutting_down:
            self.left_motor.publish(left_motor)
            self.right_motor.publish(right_motor)

        # Calculate stop cooldown for display
        stop_cooldown_remaining = None
        if self.last_stop_time is not None:
            time_since_last_stop = time.time() - self.last_stop_time
            if time_since_last_stop < AprilTagConfig.STOP_COOLDOWN:
                stop_cooldown_remaining = AprilTagConfig.STOP_COOLDOWN - time_since_last_stop
        
        # Add comprehensive debugging visualization
        self.image_processor.add_visualization_info(
            vis_image, 
            detection_results['is_curve'], 
            detection_results['curve_direction'],
            detection_results['error'], 
            steering, 
            red_light_detected, 
            len(apriltag_detections) if apriltag_detections else 0,
            self.is_stopping, 
            stop_cooldown_remaining,
            detection_results,  # Pass detection results for debugging
            debug_info          # Pass motor debug info
        )
        
        # Add detection points to visualization
        self.image_processor.add_detection_points(
            vis_image, 
            detection_results['yellow_points'], 
            detection_results['white_points'], 
            detection_results['y_points']
        )
        
        # Create contour visualization
        contour_img = self.image_processor.create_contour_visualization(
            self.image.shape, 
            detection_results['yellow_contours'], 
            detection_results['white_contours']
        )
        
        # Display both images
        cv2.imshow(self._window, vis_image)  # Show main image with debug info
        cv2.imshow("Contours", contour_img)  # Show contour detection
        cv2.waitKey(1)


if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()