#!/usr/bin/env python3

class DriveConfig:
    BASE_SPEED = 0.18
    CURVE_SPEED = 0.13
    MAX_MOTOR_VALUE = 1.0
    MIN_MOTOR_VALUE = -1.0
    CURVE_BOOST_FACTOR = 1.3

class ControlConfig:
    P_GAIN = 0.65
    D_GAIN = 0.35
    MAX_STEER = 0.4
    STEERING_THRESHOLD = 0.3

class VisionConfig:
    NEAR_FIELD_START = 0.6
    FAR_FIELD_START = 0.4
    FAR_FIELD_END = 0.6
    ROI_START = 0.55
    
    # Yellow lane detection - Enhanced HSV ranges
    YELLOW_LOWER_HSV = [15, 80, 150]
    YELLOW_UPPER_HSV = [35, 255, 255]
    YELLOW_BRIGHT_LOWER_HSV = [15, 80, 180]  # For bright yellow lines
    YELLOW_BRIGHT_UPPER_HSV = [35, 255, 255]
    
    # White lane detection - HLS
    WHITE_LOWER_HLS = [21, 0, 163]
    WHITE_UPPER_HLS = [161, 58, 203]
    
    KERNEL_SIZE = (5, 5)
    DILATE_ITERATIONS = 1
    
    CURVE_THRESHOLD = 15
    MIN_LINE_PIXELS = 500
    LINE_OFFSET = 160
    
    NUM_SLICES = 3
    SLICE_TOLERANCE = 5

class SmoothingConfig:
    SMOOTHING_STRAIGHT = 3
    SMOOTHING_CURVE = 2

class RecoveryConfig:
    RECOVERY_LEFT_SPEED = -0.2
    RECOVERY_RIGHT_SPEED = -0.3

class TrafficLightConfig:
    DETECTION_REGION_TOP = 0.0
    DETECTION_REGION_BOTTOM = 0.25
    
    RED_LOWER_1 = [0, 120, 120]
    RED_UPPER_1 = [10, 255, 255]
    RED_LOWER_2 = [170, 120, 120]
    RED_UPPER_2 = [180, 255, 255]
    
    MIN_CONTOUR_AREA = 400
    MAX_CONTOUR_AREA = 600
    MIN_ASPECT_RATIO = 0.5
    MAX_ASPECT_RATIO = 2.0
    
    MORPH_KERNEL_SIZE = (3, 3)
    ERODE_ITERATIONS = 1
    DILATE_ITERATIONS = 2
    
    DETECTION_THRESHOLD = 3
    HISTORY_SIZE = 5

class AprilTagConfig:
    STOP_SIGN_ID = 20
    MIN_TAG_SIZE = 160
    STOP_DURATION = 3.0
    STOP_COOLDOWN = 10.0  # Add this line

class ROSConfig:
    PUBLISHER_QUEUE_SIZE = 1
    CONTROL_FREQUENCY = 10