from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import pickle

from nuscenes import NuScenes
from nuscenes.utils import splits

from typing import List, Dict, Any, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import copy
NUSCENES_TRACKING_NAMES = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian'
]

class ObjectFusion:
    """
    Simplified class for fusing tracking results with detections using Hungarian matching.
    Uses detection scores to weight the fusion rather than full Kalman filtering.
    """
    
    def __init__(self, max_dist_threshold=4.0):
        """
        Initialize the ObjectFusion class.
        
        Args:
            max_dist_threshold: Maximum distance threshold for matching objects
        """
        self.max_dist_threshold = max_dist_threshold
    
    def compute_distance_matrix(self, track_objects, detection_objects):
        """
        Compute distance matrix between tracks and detections.
        
        Args:
            track_objects: List of track objects from predictions
            detection_objects: List of detection objects from noisy detections
            
        Returns:
            Distance matrix of shape [num_tracks, num_detections]
        """
        num_tracks = len(track_objects)
        num_detections = len(detection_objects)
        
        dist_matrix = np.zeros((num_tracks, num_detections))
        
        for i, track in enumerate(track_objects):
            track_pos = np.array(track['translation'][:2])  # Only use x, y for matching
            track_class = track['tracking_name']
            
            for j, det in enumerate(detection_objects):
                det_pos = np.array(det['translation'][:2])
                det_class = det['detection_name']
                
                # Euclidean distance
                dist = np.linalg.norm(track_pos - det_pos)
                
                # Apply class consistency constraint - increase distance for different classes
                if track_class != det_class:
                    dist *= 3.0
                    
                dist_matrix[i, j] = dist
                
        return dist_matrix
    
    def fuse_objects(self, tracking_results, noisy_detections):
        """
        Fuse tracking results with noisy detections using Hungarian matching.
        
        Args:
            tracking_results: List of objects from tracking_result
            noisy_detections: List of noisy detection objects
            
        Returns:
            List of fused objects
        """
        # Handle edge cases
        if not tracking_results:
            # Add tracking_id to all detections
            result = copy.deepcopy(noisy_detections)
            for i, det in enumerate(result):
                if 'tracking_id' not in det:
                    det['tracking_id'] = f'det_{i}'
                det['active'] = 1
            return result
            
        if not noisy_detections:
            # Ensure all tracking results have tracking_id
            result = copy.deepcopy(tracking_results)
            for i, track in enumerate(result):
                if 'tracking_id' not in track:
                    track['tracking_id'] = f'track_{i}'
                track['active'] = 1
            return result
        
        # Create copies to avoid modifying the inputs
        tracking_results = copy.deepcopy(tracking_results)
        noisy_detections = copy.deepcopy(noisy_detections)
        
        # Ensure all inputs have tracking_id
        for i, track in enumerate(tracking_results):
            if 'tracking_id' not in track:
                track['tracking_id'] = f'track_{i}'
        
        for i, det in enumerate(noisy_detections):
            if 'tracking_id' not in det:
                det['tracking_id'] = f'det_{i}'
        
        # Compute distance matrix for matching
        dist_matrix = self.compute_distance_matrix(tracking_results, noisy_detections)
        
        # Apply Hungarian algorithm for optimal assignment
        if dist_matrix.size > 0:
            track_indices, detection_indices = linear_sum_assignment(dist_matrix)
            
            # Filter matches by distance threshold
            matched_tracks = set()
            matched_detections = set()
            
            fused_objects = []
            
            # Process matched pairs
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if dist_matrix[track_idx, det_idx] <= self.max_dist_threshold:
                    track = tracking_results[track_idx]
                    det = noisy_detections[det_idx]
                    
                    # Mark as matched
                    matched_tracks.add(track_idx)
                    matched_detections.add(det_idx)
                    
                    # Fuse track and detection based on detection scores
                    track_score = track.get('detection_score', track.get('tracking_score', 0.5))
                    det_score = det.get('detection_score', 0.5)
                    
                    # Calculate weights based on scores
                    total_score = track_score + det_score
                    track_weight = track_score / total_score if total_score > 0 else 0.5
                    det_weight = det_score / total_score if total_score > 0 else 0.5
                    
                    # Create fused object, using weighted average
                    fused_obj = {}
                    
                    # ALWAYS SET THESE REQUIRED KEYS
                    fused_obj['active'] = 1
                    
                    # Ensure tracking_id is set (prefer track's ID)
                    fused_obj['tracking_id'] = track.get('tracking_id', det.get('tracking_id', f'fused_{len(fused_objects)}'))
                    
                    # Keep metadata from track
                    fused_obj['sample_token'] = track.get('sample_token', det.get('sample_token', ''))
                    
                    # Weighted fusion of position
                    fused_obj['translation'] = [
                        track['translation'][i] * track_weight + det['translation'][i] * det_weight
                        for i in range(len(track['translation']))
                    ]
                    
                    # Use the more confident object's size and rotation
                    if det_score > track_score:
                        fused_obj['size'] = det['size']
                        fused_obj['rotation'] = det['rotation']
                    else:
                        fused_obj['size'] = track['size']
                        fused_obj['rotation'] = track['rotation']
                    
                    # Weighted fusion of velocity if available
                    if 'velocity' in track and 'velocity' in det:
                        fused_obj['velocity'] = [
                            track['velocity'][i] * track_weight + det['velocity'][i] * det_weight
                            for i in range(min(len(track['velocity']), len(det['velocity'])))
                        ]
                    else:
                        fused_obj['velocity'] = track.get('velocity', det.get('velocity', [0, 0]))
                    
                    # Take the class from the detection if it's more confident
                    fused_obj['detection_name'] = det['detection_name'] if det_score > track_score else track.get('tracking_name', track.get('detection_name', 'car'))
                    fused_obj['tracking_name'] = fused_obj['detection_name']
                    
                    # Use max score for detection_score and tracking_score
                    fused_obj['detection_score'] = max(track_score, det_score)
                    fused_obj['tracking_score'] = fused_obj['detection_score']
                    
                    fused_objects.append(fused_obj)
            
            # Add unmatched predictions
            for i, track in enumerate(tracking_results):
                if i not in matched_tracks:
                    # Copy and add directly
                    track_copy = copy.deepcopy(track)
                    track_copy['active'] = 1
                    
                    # Ensure tracking_id is set
                    if 'tracking_id' not in track_copy:
                        track_copy['tracking_id'] = f'track_{i}'
                        
                    # Ensure detection_name is set
                    if 'detection_name' not in track_copy and 'tracking_name' in track_copy:
                        track_copy['detection_name'] = track_copy['tracking_name']
                    
                    # Ensure other required fields
                    if 'velocity' not in track_copy:
                        track_copy['velocity'] = [0, 0]
                        
                    fused_objects.append(track_copy)
            
            # Add unmatched detections
            for i, det in enumerate(noisy_detections):
                if i not in matched_detections:
                    # Convert detection to track format and add
                    det_copy = copy.deepcopy(det)
                    det_copy['tracking_name'] = det_copy['detection_name']
                    det_copy['tracking_score'] = det_copy['detection_score']
                    
                    # Ensure tracking_id is set
                    if 'tracking_id' not in det_copy:
                        det_copy['tracking_id'] = f'det_{i}'
                        
                    det_copy['active'] = 1
                    fused_objects.append(det_copy)
            
            return fused_objects
        else:
            # If dist_matrix is empty, just combine both lists
            combined = copy.deepcopy(tracking_results)
            for i, det in enumerate(noisy_detections):
                det_copy = copy.deepcopy(det)
                det_copy['tracking_name'] = det_copy.get('detection_name', 'car')
                det_copy['tracking_score'] = det_copy.get('detection_score', 0.5)
                
                # Ensure tracking_id is set
                if 'tracking_id' not in det_copy:
                    det_copy['tracking_id'] = f'det_{len(combined) + i}'
                    
                det_copy['active'] = 1
                combined.append(det_copy)
                
            return combined
    
   

def fuse_tracks_with_detections(tracking_results, noisy_detections, time_lag=None, object_fusion=None):
    """
    Wrapper function to fuse tracking results with noisy detections.
    
    Args:
        tracking_results: List of objects from tracking_result for current frame
        noisy_detections: List of noisy detection objects for current frame
        time_lag: Time difference from last frame (not used in simplified version)
        object_fusion: Optional ObjectFusion instance (will create new one if None)
        
    Returns:
        List of fused objects
    """
    if object_fusion is None:
        object_fusion = ObjectFusion()
    
    fused_objects = object_fusion.fuse_objects(tracking_results, noisy_detections)
    
    # Final safety check to ensure all required keys are present
    for i, obj in enumerate(fused_objects):
        # Required keys for the tracker
        required_keys = ['translation', 'size', 'rotation', 'velocity', 
                         'tracking_id', 'detection_name', 'detection_score']
        
        for key in required_keys:
            if key not in obj:
                # Set default values based on the key type
                if key == 'tracking_id':
                    obj[key] = f'obj_{i}'
                elif key == 'detection_name':
                    obj[key] = 'car'  # Default class
                elif key == 'detection_score':
                    obj[key] = 0.5  # Default score
                elif key == 'translation':
                    obj[key] = [0, 0, 0]  # Default position
                elif key == 'size':
                    obj[key] = [1, 1, 1]  # Default size
                elif key == 'rotation':
                    obj[key] = [1, 0, 0, 0]  # Default quaternion
                elif key == 'velocity':
                    obj[key] = [0, 0]  # Default velocity
        
        # Make sure active is set
        if 'active' not in obj:
            obj['active'] = 1
    
    return fused_objects



class ObjectListGenerator:
    """
    A class to generate noisy object lists from ground truth data.
    This simulates detection errors for testing tracker robustness.
    """
    
    def __init__(self, max_noise_std=0.1, max_drop_rate=0.1, 
                max_fp_rate=0.05, max_split_rate=0.3, seed=42):
        """
        Initialize the ObjectListGenerator class.
        
        Args:
            max_noise_std: Maximum standard deviation for position/size noise
            max_drop_rate: Maximum probability for dropping objects (false negatives)
            max_fp_rate: Maximum probability for adding false positives
            max_split_rate: Maximum probability for splitting objects or changing classes
            seed: Random seed for reproducibility
        """
        self.max_noise_std = max_noise_std
        self.max_fp_rate = max_fp_rate
        self.max_drop_rate = max_drop_rate
        self.max_split_rate = max_split_rate
        self.seed = seed

    def generate_random_box(self):
        """
        Generate a random bounding box to use as a false positive.
        
        Returns:
            Array with [x, y, z, size_x, size_y, size_z, yaw]
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        translation = np.random.uniform([-51.2, -51.2, -5.0], [51.2, 51.2, 3.0])  # Random translation
        size = np.random.uniform(0.5, 5, 3)            # Random size
        yaw = np.random.uniform(-np.pi, np.pi, 1)    # Random yaw angle
        
        return np.concatenate([translation, size, yaw])

    def apply_noise(self, bboxes):
        """
        Apply noise to position, size, and yaw of bounding boxes.
        
        Args:
            bboxes: NumPy array of bounding boxes [x, y, z, size_x, size_y, size_z, yaw]
            
        Returns:
            Array of noisy bounding boxes
        """
        current_noise_std = np.random.uniform(0, self.max_noise_std)
        
        # Calculate noise standard deviation for each axis proportional to the absolute coordinates
        noise_std = current_noise_std * np.abs(bboxes[:, :3])  # [num, 3]

        # Apply different noise levels to x, y, and z coordinates
        noise = np.random.randn(*bboxes[:, :3].shape) * noise_std

        # Apply noise to bboxes
        noisy_bboxes = bboxes.copy()
        noisy_bboxes[:, :3] += noise  # Apply noise to x, y, z coordinates

        # Introduce noise to sizes (size_x, size_y, size_z)
        size_noise = np.random.randn(*bboxes[:, 3:6].shape) * current_noise_std
        noisy_bboxes[:, 3:6] += size_noise  # Apply noise to size_x, size_y, size_z
        
        # Make sure sizes remain positive
        noisy_bboxes[:, 3:6] = np.maximum(noisy_bboxes[:, 3:6], 0.01)
        
        # Apply noise to yaw
        yaw_noise = np.random.randn(bboxes.shape[0], 1) * (current_noise_std * 0.5)
        noisy_bboxes[:, 6:7] += yaw_noise  # Apply noise to yaw
        
        # Normalize yaw to [-π, π]
        noisy_bboxes[:, 6:7] = (noisy_bboxes[:, 6:7] + np.pi) % (2 * np.pi) - np.pi

        return noisy_bboxes

    def create_noisy_detections(self, gt_boxes, gt_velocity, gt_names):
        """
        Create noisy detections from ground truth data.
        
        Args:
            gt_boxes: NumPy array of shape [N, 7] with [x, y, z, size_x, size_y, size_z, yaw]
            gt_velocity: NumPy array of shape [N, 2] with [vx, vy]
            gt_names: List of strings representing object classes
            
        Returns:
            Tuple of (noisy_boxes, noisy_velocity, noisy_names, noisy_scores)
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        if len(gt_boxes) == 0:
            return np.empty((0, 7)), np.empty((0, 2)), [], np.empty(0)
        
        # Randomize noise and rates within the specified maximum values
        current_drop_rate = np.random.uniform(0, self.max_drop_rate)
        current_false_positive_rate = np.random.uniform(0, self.max_fp_rate)
        current_split_rate = np.random.uniform(0, self.max_split_rate)

        num_objects = len(gt_boxes)
        
        # Apply noise to boxes
        noisy_boxes = self.apply_noise(gt_boxes)
        
        # Drop some true positives (create false negatives)
        keep_mask = np.random.rand(num_objects) > current_drop_rate
        noisy_boxes = noisy_boxes[keep_mask]
        noisy_velocity = gt_velocity[keep_mask] if gt_velocity is not None else None
        noisy_names = [gt_names[i] for i in range(num_objects) if keep_mask[i]]
        
        # Generate random scores (higher for true objects)
        noisy_scores = np.random.uniform(0.5, 1.0, size=len(noisy_names))
        
        # Add false positives
        num_false_positives = int(current_false_positive_rate * num_objects)
        if num_false_positives > 0:
            # Generate false positive boxes
            fp_boxes = np.zeros((num_false_positives, 7))
            for i in range(num_false_positives):
                fp_boxes[i] = self.generate_random_box()
                
            # Generate random velocities for false positives
            fp_velocity = np.random.uniform(-10, 10, (num_false_positives, 2))
            
            # Generate random class names for false positives
            fp_classes = np.random.choice([
                'car', 'pedestrian', 'bicycle', 'truck', 'bus', 'trailer',
                'motorcycle', 'barrier', 'traffic_cone', 'construction_vehicle'
            ], num_false_positives)
            
            # Generate lower confidence scores for false positives
            fp_scores = np.random.uniform(0.1, 0.6, num_false_positives)
            
            # Concatenate with noisy detections
            noisy_boxes = np.vstack([noisy_boxes, fp_boxes])
            noisy_velocity = np.vstack([noisy_velocity, fp_velocity]) if noisy_velocity is not None else fp_velocity
            noisy_names.extend(fp_classes)
            noisy_scores = np.concatenate([noisy_scores, fp_scores])
        
        # Apply class confusion and splitting
        for i in range(len(noisy_names)):
            if np.random.rand() < current_split_rate:
                # Class confusion - randomly change the class
                vehicle_classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']
                small_classes = ['pedestrian', 'bicycle', 'motorcycle', 'barrier', 'traffic_cone']
                
                if noisy_names[i] in vehicle_classes:
                    noisy_names[i] = np.random.choice(vehicle_classes)
                elif noisy_names[i] in small_classes:
                    noisy_names[i] = np.random.choice(small_classes)
                    
                # Optionally create a split detection (object detected as multiple objects)
                if np.random.rand() < 0.5:  # 50% chance to split an object that's already confused
                    # Create a split with slightly modified position
                    split_box = noisy_boxes[i].copy()
                    split_box[:3] += np.random.normal(0, self.max_noise_std * 2, 3)  # Move it a bit
                    split_box[3:6] *= np.random.uniform(0.7, 0.95, 3)  # Make it slightly smaller
                    
                    # Add the split
                    noisy_boxes = np.vstack([noisy_boxes, split_box.reshape(1, -1)])
                    
                    # Add corresponding velocity
                    if noisy_velocity is not None:
                        split_velocity = noisy_velocity[i].copy()
                        split_velocity += np.random.normal(0, 1, 2)  # Add slight velocity noise
                        noisy_velocity = np.vstack([noisy_velocity, split_velocity.reshape(1, -1)])
                    
                    # Add class and score
                    noisy_names.append(noisy_names[i])  # Same class
                    noisy_scores = np.append(noisy_scores, noisy_scores[i] * np.random.uniform(0.5, 0.9))  # Lower score
        
        # Shuffle all detections
        if len(noisy_boxes) > 0:
            shuffle_idx = np.random.permutation(len(noisy_boxes))
            noisy_boxes = noisy_boxes[shuffle_idx]
            noisy_velocity = noisy_velocity[shuffle_idx] if noisy_velocity is not None else None
            noisy_names = [noisy_names[i] for i in shuffle_idx]
            noisy_scores = noisy_scores[shuffle_idx]
            
        return noisy_boxes, noisy_velocity, noisy_names, noisy_scores
    
    def transform_to_global_frame(self, boxes, velocity, lidar2ego_translation, lidar2ego_rotation, 
                                 ego2global_translation, ego2global_rotation):
        """
        Transform boxes and velocity from LiDAR coordinate frame to global frame.
        
        Args:
            boxes: Array of boxes [N, 7] with [x, y, z, size_x, size_y, size_z, yaw]
            velocity: Array of velocities [N, 2]
            lidar2ego_translation: [3] translation vector
            lidar2ego_rotation: [4] rotation quaternion (w, x, y, z)
            ego2global_translation: [3] translation vector
            ego2global_rotation: [4] rotation quaternion (w, x, y, z)
            
        Returns:
            Tuple of (global_boxes, global_velocity)
        """
        if len(boxes) == 0:
            return boxes, velocity
        
        # Transform positions from LiDAR to ego frame
        positions = boxes[:, :3].copy()
        global_positions = np.zeros_like(positions)
        global_boxes = boxes.copy()
        
        # Step 1: Transform from LiDAR to ego frame
        for i in range(len(positions)):
            # Rotate position
            pos = positions[i]
            rotated_pos = self._rotate_point(pos, lidar2ego_rotation)
            # Translate
            ego_pos = rotated_pos + lidar2ego_translation
            
            # Step 2: Transform from ego to global frame
            # Rotate position
            rotated_global_pos = self._rotate_point(ego_pos, ego2global_rotation)
            # Translate
            global_pos = rotated_global_pos + ego2global_translation
            
            global_positions[i] = global_pos
        
        # Update boxes with global positions
        global_boxes[:, :3] = global_positions
        
        # Transform yaw angles
        # Extract yaw from boxes
        yaws = boxes[:, 6].copy()
        
        # Combine rotations (simplifying assumption: only consider rotation around z axis)
        # Convert quaternion to yaw angle
        lidar2ego_yaw = self._quaternion_to_yaw(lidar2ego_rotation)
        ego2global_yaw = self._quaternion_to_yaw(ego2global_rotation)
        
        # Add yaw rotations
        global_yaws = yaws + lidar2ego_yaw + ego2global_yaw
        # Normalize to [-π, π]
        global_yaws = (global_yaws + np.pi) % (2 * np.pi) - np.pi
        
        # Update boxes with global yaws
        global_boxes[:, 6] = global_yaws
        
        # Transform velocities
        if velocity is not None and len(velocity) > 0:
            global_velocity = velocity.copy()
            
            # Rotate velocity vectors
            for i in range(len(velocity)):
                vel = np.array([velocity[i, 0], velocity[i, 1], 0])  # Convert to 3D vector
                
                # Rotate through lidar to ego
                rotated_vel = self._rotate_vector(vel, lidar2ego_rotation)
                
                # Rotate through ego to global
                global_vel = self._rotate_vector(rotated_vel, ego2global_rotation)
                
                global_velocity[i] = global_vel[:2]  # Take just x, y components
        else:
            global_velocity = velocity
            
        return global_boxes, global_velocity
    
    def _quaternion_to_yaw(self, q):
        """
        Extract yaw angle from quaternion.
        
        Args:
            q: quaternion in [w, x, y, z] format
            
        Returns:
            yaw angle in radians
        """
        # Formula: atan2(2 * (w * z + x * y), 1 - 2 * (y^2 + z^2))
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    
    def _rotate_point(self, point, quaternion):
        """
        Rotate a point using quaternion.
        
        Args:
            point: 3D point [x, y, z]
            quaternion: rotation quaternion [w, x, y, z]
            
        Returns:
            Rotated point [x', y', z']
        """
        # Extract quaternion components
        w, x, y, z = quaternion
        
        # Prepare for quaternion math
        point_quat = np.array([0, point[0], point[1], point[2]])  # Point as quaternion [0, x, y, z]
        q = np.array([w, x, y, z])
        q_conj = np.array([w, -x, -y, -z])  # Conjugate
        
        # Rotation formula: q * point_quat * q_conj
        # First part: q * point_quat
        first_w = -q[1]*point_quat[1] - q[2]*point_quat[2] - q[3]*point_quat[3]
        first_x = q[0]*point_quat[1] + q[2]*point_quat[3] - q[3]*point_quat[2]
        first_y = q[0]*point_quat[2] - q[1]*point_quat[3] + q[3]*point_quat[1]
        first_z = q[0]*point_quat[3] + q[1]*point_quat[2] - q[2]*point_quat[1]
        
        first_result = np.array([first_w, first_x, first_y, first_z])
        
        # Second part: first_result * q_conj
        result_w = first_result[0]*q_conj[0] - first_result[1]*q_conj[1] - first_result[2]*q_conj[2] - first_result[3]*q_conj[3]
        result_x = first_result[0]*q_conj[1] + first_result[1]*q_conj[0] + first_result[2]*q_conj[3] - first_result[3]*q_conj[2]
        result_y = first_result[0]*q_conj[2] - first_result[1]*q_conj[3] + first_result[2]*q_conj[0] + first_result[3]*q_conj[1]
        result_z = first_result[0]*q_conj[3] + first_result[1]*q_conj[2] - first_result[2]*q_conj[1] + first_result[3]*q_conj[0]
        
        # Extract vector part
        return np.array([result_x, result_y, result_z])
    
    def _rotate_vector(self, vector, quaternion):
        """
        Rotate a vector using quaternion (similar to rotate_point but doesn't need translation).
        
        Args:
            vector: 3D vector [x, y, z]
            quaternion: rotation quaternion [w, x, y, z]
            
        Returns:
            Rotated vector [x', y', z']
        """
        # Same implementation as rotate_point for consistency
        return self._rotate_point(vector, quaternion)
    
    def convert_to_nusc_format(self, token, boxes, velocity, names, scores):
        """
        Convert the boxes and other data to NuScenes format for the tracker.
        
        Args:
            token: Sample token
            boxes: Array of boxes [N, 7] with [x, y, z, size_x, size_y, size_z, yaw]
            velocity: Array of velocities [N, 2]
            names: List of class names
            scores: Array of confidence scores
            
        Returns:
            List of dictionaries in NuScenes format for tracking
        """
        detections = []
        
        for i in range(len(boxes)):
            box = boxes[i]
            yaw = box[6]
            
            # Convert yaw to quaternion
            # For NuScenes, we need a quaternion in [w, x, y, z] format
            quaternion = self._yaw_to_quaternion(yaw)
            
            detection = {
                'sample_token': token,
                'translation': box[:3].tolist(),
                'size': box[3:6].tolist(),
                'rotation': quaternion.tolist(),
                'velocity': velocity[i].tolist() if velocity is not None else [0, 0],
                'detection_name': names[i],
                'detection_score': float(scores[i]),
                'attribute_name': ''
            }
            
            detections.append(detection)
            
        return detections
    
    def _yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion.
        
        Args:
            yaw: rotation angle around z-axis in radians
            
        Returns:
            quaternion as [w, x, y, z]
        """
        # For rotation around z-axis only
        return np.array([
            np.cos(yaw / 2),  # w
            0,                # x
            0,                # y
            np.sin(yaw / 2)   # z
        ])
def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results", default='./')
    parser.add_argument(
        "--checkpoint",help="the dir to checkpoint which the model read from", default='/home/jguo/StreamPETR/test/stream_petr_vov_flash_800_bs2_seq_24e/Mon_Oct_23_11_04_39_2023/pts_bbox/results_nusc.json'
    )
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--score_threshold", type=int, default=0.25)
    parser.add_argument(
        "--nuscenes_val_pkl",help="the dir to validation files", default='/home/jguo/DQTrack/mmdetection3d/data/nuscenes/mmdet3d_nuscenes_30f_infos_val.pkl'
    )
    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test 
    elif args.version == 'v1.0-mini':
        scenes = splits.mini_val 

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name'] 
        if scene_name not in scenes:
            continue 

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False 
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)


def main():
    args = parse_args()
    print('Deploy OK')
    # Initialize the ObjectListGenerator
    obj_generator = ObjectListGenerator(
        max_noise_std=0.015,
        max_drop_rate=0.05,
        max_fp_rate=0.03,
        max_split_rate=0.05,
        seed=0
    )
    # Initialize object fusion instance
    object_fusion = ObjectFusion(max_dist_threshold=4.0)

    with open(args.checkpoint, 'rb') as f:
        predictions=json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']
    val_pkl_path = args.nuscenes_val_pkl
    with open(val_pkl_path, 'rb') as f:
        val_pkl = pickle.load(f)
    val_infos = val_pkl['infos']
    sample_token_to_info = {}
    for info in val_infos:
        sample_token = info['token']
        sample_token_to_info[sample_token] = info

    print(f"Created a dictionary with {len(sample_token_to_info)} sample tokens")

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token']

        # reset tracking after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            last_time_stamp = frames[i]['timestamp']

        time_lag = (frames[i]['timestamp'] - last_time_stamp) 
        last_time_stamp = frames[i]['timestamp']

        preds = predictions[token]
        sample_info = sample_token_to_info[token]
        gt_boxes = sample_info['gt_boxes']
        gt_names = sample_info['gt_names']
        gt_velocity = sample_info['gt_velocity']
        lidar2ego_translation = sample_info['lidar2ego_translation']
        lidar2ego_rotation = sample_info['lidar2ego_rotation']
        ego2global_translation = sample_info['ego2global_translation']
        ego2global_rotation = sample_info['ego2global_rotation']
        # Generate noisy detections in LiDAR frame
        noisy_boxes, noisy_velocity, noisy_names, noisy_scores = obj_generator.create_noisy_detections(
            gt_boxes, gt_velocity, gt_names
        )
        
        # Transform from LiDAR to global world coordinate frame
        global_boxes, global_velocity = obj_generator.transform_to_global_frame(
            noisy_boxes, 
            noisy_velocity,
            lidar2ego_translation,
            lidar2ego_rotation,
            ego2global_translation,
            ego2global_rotation
        )
        
        # Convert to NuScenes format for the tracker
        noisy_detections = obj_generator.convert_to_nusc_format(
            token, global_boxes, global_velocity, noisy_names, noisy_scores
        )


        # Fuse tracking results with current detections
        outputs = fuse_tracks_with_detections(preds, noisy_detections, time_lag, object_fusion)
        annos = []

        for item in outputs:
            # Check if 'active' key exists, default to 1 if not
            if item.get('active', 0) == 0:
                continue 
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})
        print(f"Processed {i+1}/{size} frames")
        print(f"Tracking {len(annos)} objects in frame {i+1}")

    
    end = time.time()

    second = (end-start) 

    speed=size / second
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed
def filter_tracking_results(input_path, output_path=None, version="v1.0-trainval", data_root=None):
    """
    Filter tracking results to only include classes supported by NuScenes tracking evaluation.
    Also ensures all sample tokens from the dataset split are present.
    
    Args:
        input_path: Path to the original tracking_result.json file
        output_path: Path where to save the filtered results (if None, will use input_path with '_filtered' suffix)
        version: NuScenes dataset version
        data_root: Path to NuScenes dataset root
        
    Returns:
        Path to the filtered results file
    """
    import json
    import os
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    
    # Define the allowed tracking classes
    NUSCENES_TRACKING_NAMES = [
        'car', 'truck', 'bus', 'trailer',
        'motorcycle', 'bicycle', 'pedestrian'
    ]
    
    # Set default output path if not provided
    if output_path is None:
        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_filtered{ext}"
    
    # Load the original results
    with open(input_path, 'r') as f:
        tracking_results = json.load(f)
    
    # Get all sample tokens from the dataset split to ensure complete coverage
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    
    # Determine which split to use
    if version == 'v1.0-trainval':
        split_scenes = splits.val
    elif version == 'v1.0-test':
        split_scenes = splits.test
    elif version == 'v1.0-mini':
        split_scenes = splits.mini_val
    else:
        raise ValueError(f"Unknown NuScenes version: {version}")
    
    # Collect all sample tokens from the split
    all_split_tokens = set()
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        scene = nusc.get('scene', scene_token)
        if scene['name'] in split_scenes:
            all_split_tokens.add(sample['token'])
    
    print(f"Found {len(all_split_tokens)} sample tokens in the {version} split")
    
    # Track statistics for reporting
    total_objects = 0
    filtered_objects = 0
    
    # Create a filtered copy of the results
    filtered_results = {
        "results": {},
        "meta": tracking_results["meta"]
    }
    
    # Process each sample
    for sample_token, annotations in tracking_results["results"].items():
        # Filter annotations to only include supported classes
        filtered_annotations = []
        for anno in annotations:
            total_objects += 1
            if anno["tracking_name"] in NUSCENES_TRACKING_NAMES:
                filtered_annotations.append(anno)
            else:
                filtered_objects += 1
                
        # Add filtered annotations to the result
        filtered_results["results"][sample_token] = filtered_annotations
    
    # Ensure all sample tokens from the split are present
    missing_tokens = all_split_tokens - set(filtered_results["results"].keys())
    for token in missing_tokens:
        filtered_results["results"][token] = []  # Add empty list for missing tokens
    
    print(f"Added {len(missing_tokens)} missing sample tokens to ensure complete coverage")
    
    # Save filtered results
    with open(output_path, 'w') as f:
        json.dump(filtered_results, f)
    
    print(f"Filtering complete:")
    print(f"  - Total objects: {total_objects}")
    print(f"  - Filtered out: {filtered_objects} ({filtered_objects/total_objects*100:.1f}% if total > 0)")
    print(f"  - Remaining: {total_objects-filtered_objects}")
    print(f"  - Total samples in output: {len(filtered_results['results'])}")
    print(f"Filtered results saved to: {output_path}")
    
    return output_path

def eval_tracking():
    args = parse_args()
    # Get the path to the original tracking results
    original_results_path = os.path.join(args.work_dir, 'tracking_result.json')
    
    
    if args.version in ['v1.0-mini', 'v1.0-trainval']:
        # Filter the results to only include supported tracking classes
        filtered_results_path = filter_tracking_results(
            original_results_path, 
            version=args.version,
            data_root=args.data_root
        )
        eval(filtered_results_path,
            args.version,
            args.work_dir,
            args.data_root
        )
    else:
        print('Only support for v1.0-mini or v1.0-trainval')

def eval(res_path, version="v1.0-trainval", output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval 
    from nuscenes.eval.common.config import config_factory as track_configs
    eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
    
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set_map[version],
        output_dir=output_dir,
        verbose=True,
        nusc_version=version,
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def test_time():
    speeds = []
    for i in range(3):
        speeds.append(main())

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    save_first_frame()
    main()
    eval_tracking()
