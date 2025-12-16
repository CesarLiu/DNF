#!/usr/bin/env python3
import pickle
import numpy as np
import os
from tqdm import tqdm

# Path to the original pickle file
input_file = '/home/jguo/PETR/PETR/data/nuscenes/mmdet3d_nuscenes_30f_infos_val_w_obj_lidar.pkl'
# Path to save the fixed pickle file
output_file = '/home/jguo/PETR/PETR/data/nuscenes/mmdet3d_nuscenes_30f_infos_val_w_obj_lidar_fixed.pkl'

# Load the pickle file
print(f"Loading pickle file: {input_file}")
with open(input_file, 'rb') as f:
    data = pickle.load(f)

# Fix the dimension mismatch in the infos
print("Fixing dimension mismatch...")
fixed_count = 0

for i, info in enumerate(tqdm(data['infos'])):
    gt_boxes = info.get('gt_boxes', np.array([]))
    gt_names = info.get('gt_names', np.array([]))
    
    # Get the number of objects
    n_boxes = len(gt_boxes)
    
    if n_boxes == 0:
        continue  # Skip empty sample
    
    # Set valid_flag to all True (1) with the same length as gt_boxes
    info['valid_flag'] = np.ones(n_boxes, dtype=bool)
    
    # Set num_lidar_pts to 50 for all objects
    info['num_lidar_pts'] = np.ones(n_boxes, dtype=int) * 50
    
    fixed_count += 1

# Save the fixed pickle file
print(f"Saving fixed pickle file to: {output_file}")
with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print(f"Fixed {fixed_count} samples")
print(f"Fixed file saved at: {output_file}")