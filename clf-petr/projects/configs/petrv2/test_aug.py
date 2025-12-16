import mmcv
import numpy as np
import cv2
import os
import tqdm
from mmdet3d.datasets import  NuScenesDataset
import importlib
import torch
import tqdm
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import torch.nn as nn
import copy
import os
import cv2
import math
#import easydict

plugin_dir = 'projects/mmdet3d_plugin/'
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]

for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
print(_module_path)
plg_lib = importlib.import_module(_module_path)


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
point_cloud_range = [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
file_client_args = dict(backend='disk')
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

automold_transforms = [
    dict(type='add_rain', rain_type='drizzle', slant=20, drop_length=20, drop_width=1, p=0.2), ## rain_type= 'drizzle','heavy','torrential'
    dict(type='add_fog', fog_coeff=0.3 , p=0.2), 
    dict(type='add_speed', speed_coeff=0.3 , p=0.2), 
    dict(type='add_shadow', no_of_shadows=2, shadow_dimension=5, p=0.2), 
    dict(type='add_sun_flare', src_radius=200, no_of_flare_circles=8, p=0.2), 

]
ida_aug_conf = {
        # "resize_lim": (0.64, 0.75),
        # "final_dim": (576, 1024),
        "resize_lim": (0.9, 0.9),
        "final_dim": (576, 1280),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (-5.4, 5.4),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    #dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[1.0, 1.0],
            reverse_angle=True,
            training=True
            ),
    # dict(type='ResizeMultiview3D', img_scale= (1440, 800), multiscale_mode='value', keep_ratio=True),
    # dict(type='ResizeMultiview3D', img_scale= (800, 448), multiscale_mode='value', keep_ratio=True),
    # dict(type='ResizeMultiview3D', img_scale=[(1800, 640), (1800, 960)], multiscale_mopython3 de='range', keep_ratio=True),
    # dict(type='RandomFlipMultiview3D', flip_ratio=0.5),
    # dict(
    #     type='AlbuMultiview3D',
    #     transforms=albu_train_transforms,
    #     keymap={
    #         'img': 'image',
    #     },
    #     update_pad_shape=False,
    #     ),
    # dict(type='AutomoldMultiview3D', transforms=automold_transforms),

    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points','gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset = NuScenesDataset(
    data_root='./data/nuscenes/',
    ann_file='./data/nuscenes/' + 'mmdet3d_nuscenes_petr_mini_train.pkl',
    pipeline=train_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=False,
    use_valid_flag=True,
    # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    box_type_3d='LiDAR'
)

# data = dataset.__getitem__(2732)
data = dataset.__getitem__(0)
print(data.keys()) ##'img_metas', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'img'
print(data['img_metas']._data['img_shape'])
print(data['img_metas']._data['pad_shape'])
print(data['img_metas']._data['scale_factor'])
img_metas = data["img_metas"]
points = data["points"]
gt_bboxes_3d = data["gt_bboxes_3d"]
imgs = data["img"]._data

corners_3d = gt_bboxes_3d._data.corners
num_bbox = corners_3d.shape[0]
pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
center_points = gt_bboxes_3d._data.gravity_center[:, :3].detach().cpu().numpy()
center_points = np.concatenate((center_points, np.ones_like(center_points[..., :1])), -1)
gt_bbox2d = []
for i in range(len(img_metas._data['lidar2img'])):
    lidar2img_rt = copy.deepcopy(img_metas._data['lidar2img'][i])
    pts_2d = pts_4d @ lidar2img_rt.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    # pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.1, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    H, W, _ = img_metas._data['img_shape'][i]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    bbox = []
    for corner_coord in imgfov_pts_2d:
        final_coords = post_process_coords(corner_coord, imsize = (W,H))
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords
            if ((max_x - min_x) >W-100) and ((max_y - min_y)>H-100):
                continue
            bbox.append([min_x, min_y, max_x, max_y])
    gt_bbox2d.append(bbox)
mean = torch.from_numpy(img_metas._data['img_norm_cfg']['mean']).view(1, -1, 1, 1).to(imgs.device)
std = torch.from_numpy(img_metas._data['img_norm_cfg']['std']).view(1, -1, 1, 1).to(imgs.device)
imgs = imgs.clone()*std + mean
path = "./project_2dboxs_flip/"
if not os.path.exists(path):
    os.makedirs(path)
point_color = (0, 0, 255)  # BGR
thickness = 4  # 可以为 0 、4、8
if gt_bbox2d:
    for i in range(len(gt_bbox2d)):
        # img = cv2.imread(img_metas._data["filename"][i])
        img = imgs[i].clone().permute((1, 2, 0)).detach().cpu().numpy()
        cv2.imwrite(path+img_metas._data["filename"][i].split("/")[-1], img)
        img = cv2.imread(path+img_metas._data["filename"][i].split("/")[-1])
        # print(img.shape)
        for j in range(len(gt_bbox2d[i])):
            xmin,ymin,xmax,ymax = gt_bbox2d[i][j]
            # print(xmin,ymin,xmax,ymax)
            cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), point_color, thickness)
        print(path+img_metas._data["filename"][i].split("/")[-1])
        cv2.imwrite(path+img_metas._data["filename"][i].split("/")[-1], img)