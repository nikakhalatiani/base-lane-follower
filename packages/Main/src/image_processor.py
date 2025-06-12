#!/usr/bin/env python3

import cv2
import numpy as np
from config import VisionConfig, TrafficLightConfig, AprilTagConfig

class ImageProcessor:
    
    def __init__(self):
        self.kernel = np.ones(VisionConfig.KERNEL_SIZE, np.uint8)
        self.red_light_history = []
        self.red_detection_info = {}
        try:
            from pupil_apriltags import Detector
            self.apriltag_detector = Detector(
                families='tag36h11',
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
            self.apriltag_available = True
            self.apriltag_library = 'pupil'
            print("AprilTag detector (pupil-apriltags) initialized with tag36h11 family")
        except (ImportError, ModuleNotFoundError):
            try:
                import apriltag
                self.apriltag_detector = apriltag.Detector()
                self.apriltag_available = True
                self.apriltag_library = 'standard'
                print("AprilTag detector (standard apriltag) initialized")
            except (ImportError, TypeError):
                self.apriltag_available = False
                self.apriltag_library = None
                print("AprilTag library not available. System will work without AprilTag detection.")
    
    def calculate_tag_size(self, corners):
        try:
            corners = np.array(corners, dtype=np.float32)
            if corners.shape[0] < 3:
                return 0
            corners = corners.reshape((-1, 1, 2))
            perimeter = cv2.arcLength(corners, True)
            return perimeter
        except Exception as e:
            print(f"Error calculating tag size: {e}")
            return 0
    
    def check_stop_sign(self, detections):
        for detection in detections:
            if detection['id'] == AprilTagConfig.STOP_SIGN_ID:
                tag_size = self.calculate_tag_size(detection['corners'])
                if tag_size >= AprilTagConfig.MIN_TAG_SIZE:
                    return True, tag_size
        return False, 0
    
    def detect_apriltags(self, image):
        if not self.apriltag_available:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            detections = self.apriltag_detector.detect(gray)
            
            results = []
            for detection in detections:
                try:
                    if self.apriltag_library == 'pupil':
                        tag_info = {
                            'id': detection.tag_id,
                            'family': getattr(detection, 'tag_family', 'tag36h11'),
                            'center': detection.center,
                            'corners': detection.corners,
                            'hamming': getattr(detection, 'hamming', 0),
                            'goodness': getattr(detection, 'decision_margin', 0.0)
                        }
                    else:
                        tag_info = {
                            'id': detection.tag_id,
                            'family': getattr(detection, 'tag_family', 'tag36h11'),
                            'center': detection.center,
                            'corners': detection.corners,
                            'hamming': getattr(detection, 'hamming', 0),
                            'goodness': getattr(detection, 'goodness', 0.0)
                        }
                    results.append(tag_info)
                    print(f"Tag detected: ID={tag_info['id']}, Center=({tag_info['center'][0]:.1f}, {tag_info['center'][1]:.1f})")
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"AprilTag detection error: {e}")
            return []
    
    def add_apriltag_visualization(self, image, detections):
        for detection in detections:
            try:
                corners = np.array(detection['corners'], dtype=np.int32)
                
                cv2.polylines(image, [corners], True, (0, 255, 0), 2)
                
                center = np.array(detection['center'], dtype=np.int32)
                cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)
                
                cv2.putText(image, f"ID: {detection['id']}", 
                           (center[0] - 20, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error visualizing AprilTag: {e}")
                continue
    
    def preprocess_image(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def create_color_masks(self, image, slider_values=None):
        h, w = image.shape[:2]
        
        # Use HSV for yellow and HLS for white
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        # Yellow detection - use sliders if provided, otherwise config
        if slider_values and 'yellow_hsv' in slider_values:
            lb_yellow = np.array(slider_values['yellow_hsv'][0])
            ub_yellow = np.array(slider_values['yellow_hsv'][1])
        else:
            lb_yellow = np.array(VisionConfig.YELLOW_LOWER_HSV)
            ub_yellow = np.array(VisionConfig.YELLOW_UPPER_HSV)
        mask_yellow = cv2.inRange(hsv, lb_yellow, ub_yellow)
        
        # White detection - use sliders if provided, otherwise config
        if slider_values and 'white_hls' in slider_values:
            lb_white = np.array(slider_values['white_hls'][0])
            ub_white = np.array(slider_values['white_hls'][1])
        else:
            lb_white = np.array(VisionConfig.WHITE_LOWER_HLS)
            ub_white = np.array(VisionConfig.WHITE_UPPER_HLS)
        mask_white = cv2.inRange(hls, lb_white, ub_white)
        
        roi_start = int(h * VisionConfig.ROI_START)
        mask_yellow[:roi_start, :] = 0
        mask_white[:roi_start, :] = 0
        
        # Apply right-side white priority logic
        mask_white = self.apply_white_priority_filter(mask_white)
        
        return mask_yellow, mask_white
    
    def apply_white_priority_filter(self, mask_white):
        """
        Remove white lines entirely from the left side of the image.
        Only process white lines on the right side.
        """
        h, w = mask_white.shape
        center_x = w // 2
        
        # Zero out the entire left side
        mask_white[:, :center_x] = 0
        
        return mask_white
    
    def detect_red_light(self, image):
        h, w = image.shape[:2]
        
        roi_top = int(h * TrafficLightConfig.DETECTION_REGION_TOP)
        roi_bottom = int(h * TrafficLightConfig.DETECTION_REGION_BOTTOM)
        roi = image[roi_top:roi_bottom, :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        red_lower_1 = np.array(TrafficLightConfig.RED_LOWER_1)
        red_upper_1 = np.array(TrafficLightConfig.RED_UPPER_1)
        red_lower_2 = np.array(TrafficLightConfig.RED_LOWER_2)
        red_upper_2 = np.array(TrafficLightConfig.RED_UPPER_2)
        
        mask1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
        mask2 = cv2.inRange(hsv, red_lower_2, red_upper_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Count raw red pixels before morphology
        raw_red_pixels = np.count_nonzero(red_mask)
        
        kernel = np.ones(TrafficLightConfig.MORPH_KERNEL_SIZE, np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=TrafficLightConfig.ERODE_ITERATIONS)
        red_mask = cv2.dilate(red_mask, kernel, iterations=TrafficLightConfig.DILATE_ITERATIONS)
        
        # Count processed red pixels
        processed_red_pixels = np.count_nonzero(red_mask)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_detections = 0
        total_contours = len(contours)
        valid_contour_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if TrafficLightConfig.MIN_CONTOUR_AREA <= area <= TrafficLightConfig.MAX_CONTOUR_AREA:
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                if h_cont > 0:
                    aspect_ratio = w_cont / h_cont
                    if (TrafficLightConfig.MIN_ASPECT_RATIO <= aspect_ratio <= 
                        TrafficLightConfig.MAX_ASPECT_RATIO):
                        valid_detections += 1
                        valid_contour_areas.append(area)
        
        detected = valid_detections > 0
        self.red_light_history.append(detected)
        
        if len(self.red_light_history) > TrafficLightConfig.HISTORY_SIZE:
            self.red_light_history.pop(0)
        
        recent_detections = sum(self.red_light_history)
        final_detected = recent_detections >= TrafficLightConfig.DETECTION_THRESHOLD
        
        # Store detection info for visualization
        self.red_detection_info = {
            'raw_pixels': raw_red_pixels,
            'processed_pixels': processed_red_pixels,
            'total_contours': total_contours,
            'valid_detections': valid_detections,
            'valid_areas': valid_contour_areas,
            'history_score': recent_detections,
            'threshold': TrafficLightConfig.DETECTION_THRESHOLD,
            'final_detected': final_detected,
            'roi_top': roi_top,
            'roi_bottom': roi_bottom
        }
        
        return final_detected
    
    def clean_masks(self, mask_yellow, mask_white):
        mask_yellow = cv2.dilate(mask_yellow, self.kernel, 
                                iterations=VisionConfig.DILATE_ITERATIONS)
        mask_white = cv2.dilate(mask_white, self.kernel, 
                               iterations=VisionConfig.DILATE_ITERATIONS)
        
        return mask_yellow, mask_white
    
    def find_contours(self, mask_yellow, mask_white):
        yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        
        return yellow_contours, white_contours
    
    def create_contour_visualization(self, image_shape, yellow_contours, white_contours):
        contour_img = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(contour_img, yellow_contours, -1, (0, 255, 255), 2)
        cv2.drawContours(contour_img, white_contours, -1, (255, 255, 255), 2)
        return contour_img
    
    def add_visualization_info(self, image, curve_detected, curve_direction, error, steering, red_light_detected=False, apriltag_count=0, is_stopping=False, stop_cooldown_remaining=None, detection_results=None, motor_debug=None):
        cv2.putText(image, f"Curve: {curve_detected}, Dir: {curve_direction}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"Error: {error:.2f}, Steer: {steering:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add detection debugging
        if detection_results:
            left_det = "Y" if detection_results.get('left_detected') else "N"
            right_det = "Y" if detection_results.get('right_detected') else "N"
            yellow_only = detection_results.get('left_detected') and not detection_results.get('right_detected')
            cv2.putText(image, f"L:{left_det} R:{right_det} YellowOnly:{yellow_only}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show pixel counts
            yellow_pixels = detection_results.get('yellow_pixels', 0)
            white_pixels = detection_results.get('white_pixels', 0)
            cv2.putText(image, f"Y_pix:{yellow_pixels} W_pix:{white_pixels}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add motor debugging
        if motor_debug:
            cv2.putText(image, f"L_motor:{motor_debug['left']:.2f} R_motor:{motor_debug['right']:.2f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(image, f"Speed:{motor_debug['speed']:.2f} YellowMode:{motor_debug['yellow_mode']}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Red light detection details
        if hasattr(self, 'red_detection_info') and self.red_detection_info:
            info = self.red_detection_info
            # Main status
            status_color = (0, 0, 255) if red_light_detected else (0, 255, 0)
            status_text = "RED DETECTED!" if red_light_detected else "No Red"
            cv2.putText(image, f"Red: {status_text}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Pixel counts
            cv2.putText(image, f"Raw:{info['raw_pixels']} Proc:{info['processed_pixels']}", 
                       (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            
            # Contour analysis
            cv2.putText(image, f"Cont:{info['total_contours']} Valid:{info['valid_detections']}", 
                       (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            
            # History score
            cv2.putText(image, f"Score:{info['history_score']}/{info['threshold']}", 
                       (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            
            # ROI indicator
            h, w = image.shape[:2]
            roi_top = info['roi_top']
            roi_bottom = info['roi_bottom']
            cv2.rectangle(image, (0, roi_top), (w, roi_bottom), (0, 255, 255), 2)
        
        cv2.putText(image, f"AprilTags: {apriltag_count}", (10, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if is_stopping:
            cv2.putText(image, "STOP SIGN - STOPPING", (10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif stop_cooldown_remaining is not None and stop_cooldown_remaining > 0:
            cv2.putText(image, f"Stop cooldown: {stop_cooldown_remaining:.1f}s", (10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
    
    def add_detection_points(self, image, yellow_points, white_points, y_positions):
        for i, y in enumerate(y_positions):
            if i < len(yellow_points):
                cv2.circle(image, (yellow_points[i], y), 5, (0, 255, 255), -1)
            if i < len(white_points):
                cv2.circle(image, (white_points[i], y), 5, (255, 255, 255), -1)