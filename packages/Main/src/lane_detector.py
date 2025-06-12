#!/usr/bin/env python3
"""
Lane detection and analysis for autonomous navigation.
Handles high-level lane detection logic including curve detection.
"""

import numpy as np
import cv2
from config import VisionConfig
from image_processor import ImageProcessor

class LaneDetector:
    """Detects lanes and analyzes road geometry"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
    
    def detect_lines_in_slices(self, mask_yellow, mask_white, image_height):
        """
        Detect line positions across multiple horizontal slices
        
        Args:
            mask_yellow: Yellow line mask
            mask_white: White line mask
            image_height: Height of the image
            
        Returns:
            tuple: (yellow_x_points, white_x_points, y_points)
        """
        slice_height = int(image_height * 0.35 / VisionConfig.NUM_SLICES)
        start_y = int(image_height * VisionConfig.ROI_START)
        
        yellow_x_points = []
        white_x_points = []
        y_points = []
        
        for i in range(VisionConfig.NUM_SLICES):
            y = start_y + i * slice_height + slice_height // 2
            y_points.append(y)
            
            # Find yellow line x-position in this slice
            slice_yellow = mask_yellow[y-VisionConfig.SLICE_TOLERANCE:
                                     y+VisionConfig.SLICE_TOLERANCE, :]
            yellow_indices = np.where(slice_yellow > 0)[1]
            if len(yellow_indices) > 0:
                yellow_x = int(np.mean(yellow_indices))
                yellow_x_points.append(yellow_x)
            
            # Find white line x-position in this slice
            slice_white = mask_white[y-VisionConfig.SLICE_TOLERANCE:
                                   y+VisionConfig.SLICE_TOLERANCE, :]
            white_indices = np.where(slice_white > 0)[1]
            if len(white_indices) > 0:
                white_x = int(np.mean(white_indices))
                white_x_points.append(white_x)
                
        return yellow_x_points, white_x_points, y_points
    
    def detect_curve(self, yellow_x_points, white_x_points, yellow_pixels, white_pixels):
        """
        Detect if the vehicle is in a curve based on line positions and pixel ratios
        
        Args:
            yellow_x_points: List of yellow line x-coordinates
            white_x_points: List of white line x-coordinates
            yellow_pixels: Number of yellow pixels detected
            white_pixels: Number of white pixels detected
            
        Returns:
            tuple: (is_curve, curve_direction)
        """
        is_curve = False
        curve_direction = 0
        
        # Enhanced curve detection based on pixel ratio
        if yellow_pixels > 0 and white_pixels > 0:
            pixel_ratio = yellow_pixels / white_pixels
            # If yellow pixels are about 40-60% of white, likely entering/in yellow curve
            if 0.4 <= pixel_ratio <= 0.6:
                is_curve = True
                # Yellow curve bias - turn more aggressively toward yellow side
                curve_direction = -1  # Turn left toward yellow line
        
        # Original position-based curve detection (enhanced thresholds)
        if len(yellow_x_points) >= 2:
            yellow_diff = yellow_x_points[-1] - yellow_x_points[0]
            if abs(yellow_diff) > VisionConfig.CURVE_THRESHOLD * 0.7:  # More sensitive
                is_curve = True
                curve_direction += np.sign(yellow_diff)
                
        if len(white_x_points) >= 2:
            white_diff = white_x_points[-1] - white_x_points[0]
            if abs(white_diff) > VisionConfig.CURVE_THRESHOLD * 0.7:  # More sensitive
                is_curve = True
                curve_direction += np.sign(white_diff)
        
        return is_curve, curve_direction
    
    def calculate_tracking_error(self, yellow_x_points, white_x_points, 
                               left_detected, right_detected, image_width, prev_error):
        """
        Calculate the lateral tracking error based on detected lines
        
        Args:
            yellow_x_points: Yellow line x-coordinates
            white_x_points: White line x-coordinates
            left_detected: Whether left line is detected
            right_detected: Whether right line is detected
            image_width: Width of the image
            prev_error: Previous error value for fallback
            
        Returns:
            Normalized tracking error [-1, 1]
        """
        if (left_detected and right_detected and 
            len(yellow_x_points) > 0 and len(white_x_points) > 0):
            # Both lines visible - use both for guidance
            center_position = (yellow_x_points[-1] + white_x_points[-1]) / 2
            ideal_center = image_width / 2
            error = ideal_center - center_position
            
        elif left_detected and len(yellow_x_points) > 0:
            # Only left (yellow) line visible - estimate center
            error = image_width/2 - (yellow_x_points[-1] + VisionConfig.LINE_OFFSET)
            
        elif right_detected and len(white_x_points) > 0:
            # Only right (white) line visible - estimate center
            error = image_width/2 - (white_x_points[-1] - VisionConfig.LINE_OFFSET)
            
        else:
            # No lines visible - maintain last direction but be cautious
            error = prev_error
        
        # Normalize error to range [-1, 1]
        return np.clip(error / (image_width/2), -1, 1)
    
    def check_line_visibility(self, mask_yellow, mask_white):
        """
        Check if sufficient line pixels are detected for reliable tracking
        
        Args:
            mask_yellow: Yellow line mask
            mask_white: White line mask
            
        Returns:
            Whether sufficient lines are detected
        """
        line_pixels = np.count_nonzero(mask_yellow) + np.count_nonzero(mask_white)
        return line_pixels >= VisionConfig.MIN_LINE_PIXELS
    
    def process_frame(self, image, prev_error, slider_values=None):
        """
        Complete lane detection pipeline for a single frame
        
        Args:
            image: Input BGR image
            prev_error: Previous tracking error
            slider_values: Optional slider values for white line detection
            
        Returns:
            dict: Detection results containing error, curve info, and visualization data
        """
        # Preprocess image
        processed_image = self.image_processor.preprocess_image(image)
        h, w = image.shape[:2]
        
        # Create color masks
        mask_yellow, mask_white = self.image_processor.create_color_masks(processed_image, slider_values)
        
        # Clean masks
        mask_yellow, mask_white = self.image_processor.clean_masks(mask_yellow, mask_white)
        
        # Find contours
        yellow_contours, white_contours = self.image_processor.find_contours(
            mask_yellow, mask_white)
        
        # Check line detection
        left_line_detected = len(yellow_contours) > 0
        right_line_detected = len(white_contours) > 0
        
        # Detect line positions in slices
        yellow_x_points, white_x_points, y_points = self.detect_lines_in_slices(
            mask_yellow, mask_white, h)
        
        # Add pixel counts for curve detection
        yellow_pixels = np.count_nonzero(mask_yellow)
        white_pixels = np.count_nonzero(mask_white)
        
        # Detect curves with enhanced pixel ratio detection
        is_curve, curve_direction = self.detect_curve(yellow_x_points, white_x_points, yellow_pixels, white_pixels)
        
        # Calculate tracking error
        error = self.calculate_tracking_error(
            yellow_x_points, white_x_points, left_line_detected, 
            right_line_detected, w, prev_error)
        
        # Check if recovery is needed
        recovery_needed = not self.check_line_visibility(mask_yellow, mask_white)
        
        return {
            'error': error,
            'is_curve': is_curve,
            'curve_direction': curve_direction,
            'left_detected': left_line_detected,
            'right_detected': right_line_detected,
            'recovery_needed': recovery_needed,
            'yellow_points': yellow_x_points,
            'white_points': white_x_points,
            'y_points': y_points,
            'yellow_contours': yellow_contours,
            'white_contours': white_contours,
            'yellow_pixels': yellow_pixels,
            'white_pixels': white_pixels
        }