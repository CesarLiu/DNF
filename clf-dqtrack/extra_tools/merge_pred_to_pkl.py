import pickle
import json
import numpy as np
from pyquaternion import Quaternion
import random
file_path = 'mmdetection3d/data/infos/mmdet3d_nuscenes_30f_infos_val.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)
# data is a dict of dict_keys(['infos', 'metadata'])
# data['infos'] is a list of dicts
infos_data = data['infos']
# load predictions
pred_path = 'mmdetection3d/work_dirs/tracking_result_pointpillars.json'
with open(pred_path, 'rb') as f:
    pred_data = json.load(f)
    predictions = pred_data['results']
# merge predictions to data
# Track statistics
total_frames = len(infos_data)
processed_frames = 0
skipped_frames = 0
for frame in infos_data:
    # frame.keys() is dict_keys(['lidar_token', 'lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'visibility'])
    token = frame['token']
    # Check if token exists in predictions
    if token not in predictions:
        print(f"Warning: Token {token} not found in predictions, skipping frame")
        skipped_frames += 1
        continue
    
    pred = predictions[token]
    # pred is a list of dicts, filter out predictions with low scores
    # Sort predictions by detection score in descending order
    # gt_num = len(frame['gt_boxes'])
    # pred.sort(key=lambda p: p['detection_score'], reverse=True)
    # # Set ratios for dropping GT and introducing false positives
    # drop_ratio = 0.2  # 20% chance to drop GT objects

    # # Randomly decide how many objects to keep based on the drop ratio
    # total_select = max(1, int(gt_num * (1 - random.uniform(-drop_ratio, drop_ratio))))
 

    # # Make sure we don't try to select more than available
    # total_select = min(total_select, len(pred))

    # # Select the top predictions based on the calculated number
    # pred_filtered = pred[:total_select]
    pred_filtered = [p for p in pred if p['tracking_score'] > 0.3]
    if len(pred_filtered) == 0:
        print(f"Warning: No predictions above threshold for token {token}, skipping frame")
        skipped_frames += 1
        continue
    
    # Get transformation matrices for this frame
    lidar2ego_trans = np.array(frame['lidar2ego_translation'])
    lidar2ego_rot = Quaternion(frame['lidar2ego_rotation'])
    ego2global_trans = np.array(frame['ego2global_translation'])
    ego2global_rot = Quaternion(frame['ego2global_rotation'])
    
    # Combined transformation: global to lidar
    global2ego_rot = ego2global_rot.inverse
    ego2lidar_rot = lidar2ego_rot.inverse
    global2lidar_rot = ego2lidar_rot * global2ego_rot
    
    # Process each prediction and transform to lidar frame
    transformed_translations = []
    transformed_sizes = []
    transformed_rotations = []
    
    for p in pred_filtered:
        # Get object's position and orientation in global frame
        global_translation = np.array(p['translation'])
        global_rotation = Quaternion(p['rotation'])
        obj_size = np.array(p['size'])
        
        # Step 1: Transform from global to ego frame
        ego_translation = global2ego_rot.rotate(global_translation - ego2global_trans)
        ego_rotation = global2ego_rot * global_rotation
        
        # Step 2: Transform from ego to lidar frame
        lidar_translation = ego2lidar_rot.rotate(ego_translation - lidar2ego_trans)
        lidar_rotation = ego2lidar_rot * ego_rotation
        
        # Get yaw angle in lidar frame
        lidar_yaw = lidar_rotation.yaw_pitch_roll[0]
        
        # Store transformed values
        transformed_translations.append(lidar_translation)
        transformed_sizes.append(obj_size)
        transformed_rotations.append(lidar_yaw)
    
    # Convert lists to numpy arrays
    translation = np.array(transformed_translations)
    size = np.array(transformed_sizes)
    rotation = np.array(transformed_rotations)
    rotation = np.expand_dims(rotation, axis=1)
    
    # Apply standard rotation adjustment for NuScenes
    rotation_adjusted = -rotation - np.pi / 2
    
    # Concatenate to create the bounding box array in lidar coordinate system
    frame['gt_boxes'] = np.concatenate([translation, size, rotation_adjusted], axis=1)
    frame['gt_names'] = np.array([p['tracking_name'] for p in pred_filtered])
    frame['gt_velocity'] = np.array([p['velocity'] for p in pred_filtered])
    # Set default values
    frame['num_lidar_pts'] = 50*np.ones(len(pred_filtered), dtype=bool)
    frame['valid_flag'] = np.ones(len(pred_filtered), dtype=bool)
    processed_frames += 1
print(f"Processing summary:")
print(f"Total frames: {total_frames}")
print(f"Processed frames: {processed_frames}")
print(f"Skipped frames: {skipped_frames}")
new_pkl_path = 'mmdetection3d/data/infos/mmdet3d_nuscenes_30f_infos_val_with_pointpillars.pkl'
# save the merged data
data['infos'] = infos_data
with open(new_pkl_path, 'wb') as f:
    pickle.dump(data, f)

