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
from config import AprilTagConfig

BASE_SPEED = 0.25
CURVE_SPEED = 0.20
P_GAIN = 0.4
D_GAIN = 0.2
MAX_STEER = 0.5
SMOOTHING_STRAIGHT = 3
SMOOTHING_CURVE = 2

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.base_speed = BASE_SPEED
        self.curve_speed = CURVE_SPEED
        self.p_gain = P_GAIN
        self.d_gain = D_GAIN
        self.max_steer = MAX_STEER
        
        self.prev_error = 0
        self.left_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        self.right_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        
        self.image_processor = ImageProcessor()
        
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
        # Create trackbars for white color adjustment
        cv2.createTrackbar('Lower Hue', self._window, 21, 180, lambda x: None)
        cv2.createTrackbar('Upper Hue', self._window, 180, 180, lambda x: None)
        cv2.createTrackbar('Lower Saturation', self._window, 0, 255, lambda x: None)
        cv2.createTrackbar('Upper Saturation', self._window, 67, 255, lambda x: None)
        cv2.createTrackbar('Lower Value', self._window, 164, 255, lambda x: None)
        cv2.createTrackbar('Upper Value', self._window, 232, 255, lambda x: None)

        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self):
        self.shutting_down = True
        self.left_motor.publish(0)
        self.right_motor.publish(0)
        cv2.destroyAllWindows()
        
    def smooth_motor_value(self, value, history_buffer):
        history_buffer.append(value)
        return sum(history_buffer) / len(history_buffer)

    def callback(self, msg):
        if self.shutting_down:
            return

        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.image = cv2.bilateralFilter(self.image, 9, 75, 75)

        vis_image = self.image.copy()
        
        red_light_detected = self.image_processor.detect_red_light(self.image)
        
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
        
        h, w = self.image.shape[:2]
        near_field = self.image[int(h*0.6):, :]
        far_field = self.image[int(h*0.4):int(h*0.6), :]
        
        luv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lb_yellow = np.array([15, 80, 150])
        ub_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(luv, lb_yellow, ub_yellow)
        mask_yellow[:int(h*0.55), :] = 0
        
        
        lower_hue = cv2.getTrackbarPos('Lower Hue', self._window)
        upper_hue = cv2.getTrackbarPos('Upper Hue', self._window)
        lower_saturation = cv2.getTrackbarPos('Lower Saturation', self._window)
        upper_saturation = cv2.getTrackbarPos('Upper Saturation', self._window)
        lower_value = cv2.getTrackbarPos('Lower Value', self._window)
        upper_value = cv2.getTrackbarPos('Upper Value', self._window)


        lb_white = np.array([lower_hue, lower_saturation, lower_value])
        ub_white = np.array([upper_hue, upper_saturation, upper_value])
        mask_white = cv2.inRange(hls, lb_white, ub_white)
        mask_white[:int(h*0.55), :] = 0
        
        kernel = np.ones((5, 5), np.uint8)
        mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
        mask_white = cv2.dilate(mask_white, kernel, iterations=1)
        
        yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_img = np.zeros_like(self.image)
        cv2.drawContours(contour_img, yellow_contours, -1, (0, 255, 255), 2)
        cv2.drawContours(contour_img, white_contours, -1, (255, 255, 255), 2)
        
        left_line_detected = len(yellow_contours) > 0
        right_line_detected = len(white_contours) > 0
        
        num_slices = 3
        slice_height = int(h * 0.35 / num_slices)
        start_y = int(h * 0.55)
        
        yellow_x_points = []
        white_x_points = []
        y_points = []
        
        for i in range(num_slices):
            y = start_y + i * slice_height + slice_height // 2
            y_points.append(y)
            
            slice_yellow = mask_yellow[y-5:y+5, :]
            yellow_indices = np.where(slice_yellow > 0)[1]
            if len(yellow_indices) > 0:
                yellow_x = int(np.mean(yellow_indices))
                yellow_x_points.append(yellow_x)
                cv2.circle(vis_image, (yellow_x, y), 5, (0, 255, 255), -1)
            
            slice_white = mask_white[y-5:y+5, :]
            white_indices = np.where(slice_white > 0)[1]
            if len(white_indices) > 0:
                white_x = int(np.mean(white_indices))
                white_x_points.append(white_x)
                cv2.circle(vis_image, (white_x, y), 5, (255, 255, 255), -1)
                
        is_curve = False
        curve_direction = 0
        
        if len(yellow_x_points) >= 2:
            yellow_diff = yellow_x_points[-1] - yellow_x_points[0]
            if abs(yellow_diff) > 20:
                is_curve = True
                curve_direction += np.sign(yellow_diff)
                
        if len(white_x_points) >= 2:
            white_diff = white_x_points[-1] - white_x_points[0]
            if abs(white_diff) > 20:
                is_curve = True
                curve_direction += np.sign(white_diff)
        
        if left_line_detected and right_line_detected and len(yellow_x_points) > 0 and len(white_x_points) > 0:
            center_position = (yellow_x_points[-1] + white_x_points[-1]) / 2
            ideal_center = w / 2
            error = ideal_center - center_position
        elif left_line_detected and len(yellow_x_points) > 0:
            error = w/2 - (yellow_x_points[-1] + 160)
        elif right_line_detected and len(white_x_points) > 0:
            error = w/2 - (white_x_points[-1] - 160)
        else:
            error = self.prev_error
        
        error = np.clip(error / (w/2), -1, 1)
        
        error_diff = error - self.prev_error
        self.prev_error = error
        
        steering = self.p_gain * error + self.d_gain * error_diff
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        
        current_speed = self.curve_speed if is_curve else self.base_speed
        
        left_motor = current_speed - steering
        right_motor = current_speed + steering
        
        if is_curve and abs(steering) > 0.3:
            if steering > 0:
                right_motor *= 1.3
            else:
                left_motor *= 1.3
        
        line_pixels = np.count_nonzero(mask_yellow) + np.count_nonzero(mask_white)
        if line_pixels < 500:
            left_motor = -0.2
            right_motor = -0.3
            
        smoothing_amount = SMOOTHING_CURVE if is_curve else SMOOTHING_STRAIGHT
        self.left_motor_history = deque(self.left_motor_history, maxlen=smoothing_amount)
        self.right_motor_history = deque(self.right_motor_history, maxlen=smoothing_amount)
        
        left_motor = self.smooth_motor_value(left_motor, self.left_motor_history)
        right_motor = self.smooth_motor_value(right_motor, self.right_motor_history)
        
        left_motor = np.clip(left_motor, -1.0, 1.0)
        right_motor = np.clip(right_motor, -1.0, 1.0)
        
        if self.is_stopping:
            left_motor = 0.0
            right_motor = 0.0
        elif red_light_detected:
            left_motor = 0.0
            right_motor = 0.0
        
        if not self.shutting_down:
            self.left_motor.publish(left_motor)
            self.right_motor.publish(right_motor)

        stop_cooldown_remaining = None
        if self.last_stop_time is not None:
            time_since_last_stop = time.time() - self.last_stop_time
            if time_since_last_stop < AprilTagConfig.STOP_COOLDOWN:
                stop_cooldown_remaining = AprilTagConfig.STOP_COOLDOWN - time_since_last_stop
        
        self.image_processor.add_visualization_info(vis_image, is_curve, curve_direction, 
                                                   error, steering, red_light_detected, 
                                                   len(apriltag_detections) if apriltag_detections else 0,
                                                   self.is_stopping, stop_cooldown_remaining)
        
        cv2.imshow(self._window, contour_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()