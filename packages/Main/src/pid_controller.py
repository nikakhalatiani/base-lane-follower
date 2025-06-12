#!/usr/bin/env python3
"""
PID/PD controller for lane following and motor control.
Handles steering calculations and motor command generation.
"""

import numpy as np
from collections import deque
from config import ControlConfig, DriveConfig, SmoothingConfig, RecoveryConfig

class PDController:
    """PD controller for lane following with adaptive smoothing"""
    
    def __init__(self):
        self.prev_error = 0
        self.left_motor_history = deque(maxlen=SmoothingConfig.SMOOTHING_STRAIGHT)
        self.right_motor_history = deque(maxlen=SmoothingConfig.SMOOTHING_STRAIGHT)
    
    def calculate_steering(self, error):
        """
        Calculate steering command using PD control
        
        Args:
            error: Normalized tracking error [-1, 1]
            
        Returns:
            Steering command [-MAX_STEER, MAX_STEER]
        """
        # Calculate derivative of error for D-term
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # PD control
        steering = ControlConfig.P_GAIN * error + ControlConfig.D_GAIN * error_diff
        return np.clip(steering, -ControlConfig.MAX_STEER, ControlConfig.MAX_STEER)
    
    def calculate_motor_commands(self, steering, is_curve, recovery_needed=False, yellow_only_mode=False, speed_override=None):
        """
        Calculate left and right motor commands
        
        Args:
            steering: Steering command from PD controller
            is_curve: Whether vehicle is in a curve
            recovery_needed: Whether recovery behavior is needed
            yellow_only_mode: Whether only yellow line is detected (more aggressive)
            speed_override: Override speed setting from slider
            
        Returns:
            tuple: (left_motor, right_motor) commands
        """
        if recovery_needed:
            return RecoveryConfig.RECOVERY_LEFT_SPEED, RecoveryConfig.RECOVERY_RIGHT_SPEED
        
        # Use speed override or default speeds
        if speed_override is not None:
            base_speed = speed_override
            curve_speed = speed_override * 0.72  # Maintain 72% ratio for curves
        else:
            base_speed = DriveConfig.BASE_SPEED
            curve_speed = DriveConfig.CURVE_SPEED
        
        # Adjust speeds based on curve detection
        current_speed = curve_speed if is_curve else base_speed
        
        # Yellow-only mode: increase steering response significantly
        if yellow_only_mode:
            steering *= 2.0  # Double the steering response when only yellow is detected
            current_speed *= 0.8  # Slightly reduce speed for better control
        
        # Calculate basic motor values
        left_motor = current_speed - steering
        right_motor = current_speed + steering
        
        # Backwards motion compensation: accelerate right wheel more
        if left_motor < 0 and right_motor < 0:  # Both wheels going backwards
            right_motor *= 1.15  # 15% more power to right wheel when reversing
        
        # Special case: if in a tight curve, help by differential steering
        if is_curve and abs(steering) > ControlConfig.STEERING_THRESHOLD:
            # Boost the inside wheel in curves
            if steering > 0:  # Turning left
                right_motor *= DriveConfig.CURVE_BOOST_FACTOR
            else:  # Turning right
                left_motor *= DriveConfig.CURVE_BOOST_FACTOR
        
        return left_motor, right_motor
    
    def smooth_motor_value(self, value, history_buffer):
        """
        Apply smoothing to motor values using moving average
        
        Args:
            value: Current motor value
            history_buffer: Deque containing history of values
            
        Returns:
            Smoothed motor value
        """
        history_buffer.append(value)
        return sum(history_buffer) / len(history_buffer)
    
    def apply_smoothing(self, left_motor, right_motor, is_curve, yellow_only_mode=False):
        """
        Apply adaptive smoothing to motor commands
        
        Args:
            left_motor: Raw left motor command
            right_motor: Raw right motor command
            is_curve: Whether vehicle is in a curve
            yellow_only_mode: Whether only yellow line is detected (more aggressive)
            
        Returns:
            tuple: (smoothed_left, smoothed_right) motor commands
        """
        # Determine smoothing amount based on detection mode and curve
        if yellow_only_mode:
            smoothing_amount = 1  # Minimal smoothing for aggressive yellow response
        else:
            smoothing_amount = (SmoothingConfig.SMOOTHING_CURVE if is_curve 
                              else SmoothingConfig.SMOOTHING_STRAIGHT)
        
        # Update buffer sizes
        self.left_motor_history = deque(self.left_motor_history, maxlen=smoothing_amount)
        self.right_motor_history = deque(self.right_motor_history, maxlen=smoothing_amount)
        
        # Apply smoothing
        left_smoothed = self.smooth_motor_value(left_motor, self.left_motor_history)
        right_smoothed = self.smooth_motor_value(right_motor, self.right_motor_history)
        
        return left_smoothed, right_smoothed
    
    def apply_safety_bounds(self, left_motor, right_motor):
        """
        Apply safety bounds to motor commands
        
        Args:
            left_motor: Left motor command
            right_motor: Right motor command
            
        Returns:
            tuple: (bounded_left, bounded_right) motor commands
        """
        left_motor = np.clip(left_motor, DriveConfig.MIN_MOTOR_VALUE, 
                           DriveConfig.MAX_MOTOR_VALUE)
        right_motor = np.clip(right_motor, DriveConfig.MIN_MOTOR_VALUE, 
                            DriveConfig.MAX_MOTOR_VALUE)
        return left_motor, right_motor
    
    def process_control_loop(self, detection_results, speed_override=None):
        """
        Complete control loop processing
        
        Args:
            detection_results: Results from lane detection
            speed_override: Optional speed override from slider
            
        Returns:
            tuple: (left_motor, right_motor, steering, debug_info) final commands
        """
        # Calculate steering
        steering = self.calculate_steering(detection_results['error'])
        
        # Check if only yellow line is detected (more aggressive mode)
        yellow_only_mode = (detection_results['left_detected'] and 
                          not detection_results['right_detected'])
        
        # Calculate motor commands
        left_motor, right_motor = self.calculate_motor_commands(
            steering, detection_results['is_curve'], detection_results['recovery_needed'], yellow_only_mode, speed_override)
        
        # Store current speed for debugging
        if speed_override is not None:
            base_speed = speed_override
            curve_speed = speed_override * 0.72
        else:
            base_speed = DriveConfig.BASE_SPEED
            curve_speed = DriveConfig.CURVE_SPEED
            
        current_speed = curve_speed if detection_results['is_curve'] else base_speed
        if yellow_only_mode:
            current_speed *= 0.8
        
        # Apply adaptive smoothing
        left_motor, right_motor = self.apply_smoothing(
            left_motor, right_motor, detection_results['is_curve'], yellow_only_mode)
        
        # Apply safety bounds
        left_motor, right_motor = self.apply_safety_bounds(left_motor, right_motor)
        
        # Create debug info
        debug_info = {
            'left': left_motor,
            'right': right_motor,
            'speed': current_speed,
            'yellow_mode': yellow_only_mode,
            'raw_steering': steering * (2.0 if yellow_only_mode else 1.0)
        }
        
        return left_motor, right_motor, steering, debug_info