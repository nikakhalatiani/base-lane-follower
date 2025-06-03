#!/usr/bin/env python3
import os
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
        
        h, w = self.image.shape[:2]
        near_field = self.image[int(h*0.6):, :]
        far_field = self.image[int(h*0.4):int(h*0.6), :]
        
        luv = cv2.cvtColor(self.image, cv2.COLOR_BGR2LUV)
        hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)

        lb_yellow = np.array([10, 85, 160])
        ub_yellow = np.array([255, 255, 255])
        mask_yellow = cv2.inRange(luv, lb_yellow, ub_yellow)
        mask_yellow[:int(h*0.55), :] = 0
        
        lb_white = np.array([0, 144, 0])
        ub_white = np.array([168, 255, 36])
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
        
        if red_light_detected:
            left_motor = 0.0
            right_motor = 0.0
        
        if not self.shutting_down:
            self.left_motor.publish(left_motor)
            self.right_motor.publish(right_motor)

        self.image_processor.add_visualization_info(vis_image, is_curve, curve_direction, 
                                                   error, steering, red_light_detected)
        
        cv2.imshow(self._window, vis_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()