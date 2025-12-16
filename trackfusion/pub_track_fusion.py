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
import copy
import json
import os
import numpy as np
from pub_tracker import PubTracker as Tracker
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits
from scipy.optimize import linear_sum_assignment

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results", default='./')
    parser.add_argument(
        "--checkpoint1",help="the dir to checkpoint1 which the model read from", default='./results_nusc.json'
    )
    parser.add_argument(
        "--checkpoint2",help="the dir to checkpoint2 which the model read from", default='./results_nusc.json'
    )
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--score_threshold", type=int, default=0.25)
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


def fuse_tracking_results(tracks1, tracks2, distance_threshold=2.0, score_weight=0.7):
    """
    Fuse tracking results from two models by matching tracks and combining them.
    
    Args:
        tracks1: List of tracks from model 1
        tracks2: List of tracks from model 2  
        distance_threshold: Maximum distance to consider tracks as the same object
        score_weight: Weight for combining detection scores
    
    Returns:
        List of fused tracking results
    """
    if not tracks1 and not tracks2:
        return []
    if not tracks1:
        return tracks2
    if not tracks2:
        return tracks1
    
    # Extract positions and create cost matrix
    pos1 = np.array([track['translation'][:2] for track in tracks1])
    pos2 = np.array([track['translation'][:2] for track in tracks2])
    
    # Calculate distance matrix
    dist_matrix = np.linalg.norm(pos1[:, np.newaxis] - pos2[np.newaxis, :], axis=2)
    
    # Create class compatibility matrix
    class1 = [track['tracking_name'] for track in tracks1]
    class2 = [track['tracking_name'] for track in tracks2]
    class_matrix = np.array([[c1 != c2 for c2 in class2] for c1 in class1])
    
    # Set invalid matches to large distance
    cost_matrix = dist_matrix.copy()
    cost_matrix[class_matrix] = 1e6
    cost_matrix[dist_matrix > distance_threshold] = 1e6
    
    # Hungarian assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    fused_tracks = []
    matched_tracks1 = set()
    matched_tracks2 = set()
    
    # Process matched tracks
    for i, j in zip(row_indices, col_indices):
        if cost_matrix[i, j] < 1e6:  # Valid match
            track1 = tracks1[i]
            track2 = tracks2[j]
            
            # Fuse the matched tracks
            fused_track = fuse_single_track_pair(track1, track2, score_weight)
            fused_tracks.append(fused_track)
            
            matched_tracks1.add(i)
            matched_tracks2.add(j)
    
    # Add unmatched tracks from model 1
    for i, track in enumerate(tracks1):
        if i not in matched_tracks1:
            fused_tracks.append(track)
    
    # Add unmatched tracks from model 2
    for j, track in enumerate(tracks2):
        if j not in matched_tracks2:
            fused_tracks.append(track)
    
    return fused_tracks

def fuse_single_track_pair(track1, track2, score_weight=0.7):
    """
    Fuse two matched tracks by combining their properties.
    """
    # Choose the track with higher confidence as base
    if track1['tracking_score'] >= track2['tracking_score']:
        base_track = copy.deepcopy(track1)
        other_track = track2
    else:
        base_track = copy.deepcopy(track2)
        other_track = track1
    
    # Weighted average of positions
    pos1 = np.array(track1['translation'])
    pos2 = np.array(track2['translation'])
    w1 = track1['tracking_score']
    w2 = track2['tracking_score']
    
    fused_pos = (w1 * pos1 + w2 * pos2) / (w1 + w2)
    base_track['translation'] = fused_pos.tolist()
    
    # Weighted average of velocities
    vel1 = np.array(track1['velocity'])
    vel2 = np.array(track2['velocity'])
    fused_vel = (w1 * vel1 + w2 * vel2) / (w1 + w2)
    base_track['velocity'] = fused_vel.tolist()
    
    # Average the scores
    base_track['tracking_score'] = (track1['tracking_score'] + track2['tracking_score']) / 2.0
    
    # Use the size from higher confidence track
    # base_track keeps its size already
    
    return base_track


def main():
    args = parse_args()
    print('Deploy OK')

    # Load tracking results from both models (already tracked objects)
    with open(args.checkpoint1, 'rb') as f:
        tracking_results1=json.load(f)['results']

    with open(args.checkpoint2, 'rb') as f:
        tracking_results2=json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    print("Begin Tracking Fusion\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token']

        # Get tracking results from both models for this frame
        tracks1 = tracking_results1.get(token, [])
        tracks2 = tracking_results2.get(token, [])
        
        # Fuse the tracking results
        fused_tracks = fuse_tracking_results(tracks1, tracks2, 
                                           distance_threshold=2.0, 
                                           score_weight=0.7)
        
        # Convert to nuScenes format (already in correct format, just ensure consistency)
        annos = []
        for track in fused_tracks:
            nusc_anno = {
                "sample_token": token,
                "translation": track['translation'],
                "size": track['size'],
                "rotation": track['rotation'],
                "velocity": track['velocity'],
                "tracking_id": track['tracking_id'],
                "tracking_name": track['tracking_name'],
                "tracking_score": track['tracking_score'],
            }
            annos.append(nusc_anno)
        
        nusc_annos["results"].update({token: annos})

    
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

    with open(os.path.join(args.work_dir, 'fused_tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking():
    args = parse_args()
    if args.version in ['v1.0-mini', 'v1.0-trainval']:
        eval(os.path.join(args.work_dir, 'fused_tracking_result.json'),
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
