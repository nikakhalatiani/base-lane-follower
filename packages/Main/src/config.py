#!/usr/bin/env python3
"""
Configuration parameters for the lane following system.
Centralized location for all tunable parameters.
"""

class DriveConfig:
    """Drive control parameters"""
    BASE_SPEED = 0.25       # Base forward speed
    CURVE_SPEED = 0.20      # Reduced speed for curves
    MAX_MOTOR_VALUE = 1.0   # Maximum motor command value
    MIN_MOTOR_VALUE = -1.0  # Minimum motor command value
    CURVE_BOOST_FACTOR = 1.3  # Speed boost for inside wheel in curves

class ControlConfig:
    """PID/PD control parameters"""
    P_GAIN = 0.4           # Proportional gain
    D_GAIN = 0.2           # Derivative gain - helps with curves
    MAX_STEER = 0.5        # Maximum steering adjustment
    STEERING_THRESHOLD = 0.3  # Threshold for tight curve detection

class VisionConfig:
    """Computer vision parameters"""
    # Image regions of interest
    NEAR_FIELD_START = 0.6    # Bottom 40% of image
    FAR_FIELD_START = 0.4     # Middle region start
    FAR_FIELD_END = 0.6       # Middle region end
    ROI_START = 0.55          # Region of interest start for line detection
    
    # Color thresholds for line detection
    YELLOW_LOWER_LUV = [10, 85, 160]    # Yellow line lower bound in LUV
    YELLOW_UPPER_LUV = [255, 255, 255]  # Yellow line upper bound in LUV
    WHITE_LOWER_HLS = [0, 144, 0]       # White line lower bound in HLS
    WHITE_UPPER_HLS = [168, 255, 36]    # White line upper bound in HLS
    
    # Morphological operations
    KERNEL_SIZE = (5, 5)      # Morphological kernel size
    DILATE_ITERATIONS = 1     # Dilation iterations
    
    # Line detection parameters
    CURVE_THRESHOLD = 20      # Pixel difference threshold for curve detection
    MIN_LINE_PIXELS = 500     # Minimum pixels required for line detection
    LINE_OFFSET = 160         # Expected line offset from center
    
    # Image slicing for curve detection
    NUM_SLICES = 3            # Number of horizontal slices
    SLICE_TOLERANCE = 5       # Pixels above/below slice center

class SmoothingConfig:
    """Motor command smoothing parameters"""
    SMOOTHING_STRAIGHT = 3    # Smoothing for straight roads
    SMOOTHING_CURVE = 2       # Less smoothing for curves

class RecoveryConfig:
    """Recovery behavior parameters"""
    RECOVERY_LEFT_SPEED = -0.2   # Left motor speed during recovery
    RECOVERY_RIGHT_SPEED = -0.3  # Right motor speed during recovery

class TrafficLightConfig:
    """Traffic light detection parameters"""
    # Region of interest for traffic light detection
    DETECTION_REGION_TOP = 0.0      # Top of detection region (0.0 = top of image)
    DETECTION_REGION_BOTTOM = 0.25  # Bottom of detection region (adjustable percentage)
    
    # Red color detection in HSV space (handles red wrap-around)
    RED_LOWER_1 = [0, 120, 120]      # Lower red range 1
    RED_UPPER_1 = [10, 255, 255]     # Upper red range 1
    RED_LOWER_2 = [170, 120, 120]    # Lower red range 2 (wrap-around)
    RED_UPPER_2 = [180, 255, 255]    # Upper red range 2 (wrap-around)
    
    # Filtering parameters
    MIN_CONTOUR_AREA = 300           # Minimum contour area for valid detection
    MAX_CONTOUR_AREA = 5000          # Maximum contour area for valid detection
    MIN_ASPECT_RATIO = 0.5           # Minimum width/height ratio
    MAX_ASPECT_RATIO = 2.0           # Maximum width/height ratio
    
    # Morphological operations
    MORPH_KERNEL_SIZE = (3, 3)       # Kernel for morphological operations
    ERODE_ITERATIONS = 1             # Erosion iterations to remove noise
    DILATE_ITERATIONS = 2            # Dilation iterations to fill gaps
    
    # Detection stability
    DETECTION_THRESHOLD = 3          # Minimum detections in recent frames
    HISTORY_SIZE = 5                 # Number of recent frames to consider

class ROSConfig:
    """ROS-specific configuration"""
    PUBLISHER_QUEUE_SIZE = 1     # ROS publisher queue size
    CONTROL_FREQUENCY = 10       # Control loop frequency in Hz