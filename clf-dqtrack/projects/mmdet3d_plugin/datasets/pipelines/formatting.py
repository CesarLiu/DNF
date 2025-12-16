# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from UVTR (https://github.com/dvlab-research/UVTR)
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.points import BasePoints
from mmdet3d.datasets.pipelines import DefaultFormatBundle
from copy import deepcopy
from mmdet3d.core.bbox import LiDARInstance3DBoxes, BaseInstance3DBoxes
import random

@PIPELINES.register_module()
class FormatBundle3DTrack(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, with_gt=True, with_label=True):
        super(FormatBundle3DTrack, self).__init__()
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            points_cat = []
            for point in results['points']:
                assert isinstance(point, BasePoints)
                points_cat.append(point.tensor)
            # results['points'] = DC(torch.stack(points_cat, dim=0))
            results['points'] = DC(points_cat)

        if 'img' in results:
            imgs_list = results['img']
            imgs_cat_list = []
            for imgs_frame in imgs_list:
                imgs = [img.transpose(2, 0, 1) for img in imgs_frame]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                imgs_cat_list.append(to_tensor(imgs))
            
            results['img'] = DC(torch.stack(imgs_cat_list, dim=0), stack=True)
        
        if 'depth_map' in results:
            depth_cat = []
            for depth in results['depth_map']:
                depth_cat.append(torch.stack(depth))
            
            results['depth_map'] = DC(torch.stack(depth_cat, dim=0), stack=True)
            
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths',
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        
        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
        
        if 'instance_inds' in results:
            instance_inds = [torch.tensor(_t) for _t in results['instance_inds']]
            results['instance_inds'] = DC(instance_inds)
        
        keys = ['l2g_r_mat', 'l2g_t']
        for key in keys:
            if key in results:
                results[key] = DC(to_tensor(np.array(results[key])))

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str


@PIPELINES.register_module()
class CollectUnified3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 
                            'lidar2img', 'lidar2ego', 'lidar2global', 'l2g_r_mat', 'l2g_t',
                            'depth2img', 'cam2img', 'pad_shape', 'cam_intrinsic',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'sweeps_paths', 'sweeps_ids', 
                            'sweeps_time', 'uni_rot_aug', 'uni_trans_aug', 'uni_flip_aug',
                            'img_rot_aug', 'img_trans_aug', 'img_aug_mat', 
                            'rot_degree', 'scene_token', 'timestamp',
                            'intrinsics', 'extrinsics', 'lidar_timestamp',
                            'gt_bboxes_3d', 'gt_labels_3d')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'
    
@PIPELINES.register_module()
class CreateObjectListWithNoise_forobjectlist(object):
    def __init__(self, with_object_list=True, offset_scale=2.0, change_prob=0.3, drop_prob=0.2):
        self.with_object_list = with_object_list
        self.offset_scale = offset_scale
        self.change_prob = change_prob
        self.drop_prob = drop_prob

    def __call__(self, results):
        # 提取 gt_bboxes
        bboxes1 = deepcopy(results['gt_bboxes_3d']._data.gravity_center)#[num,3]
        bboxes2 = deepcopy(results['gt_bboxes_3d']._data.tensor[:,3:])#[num,6]
        bboxes_label = deepcopy(results['gt_labels_3d']._data)#[num,]
        bboxes = torch.cat((bboxes1, bboxes2,bboxes_label.unsqueeze(1)), dim=1)#[num,10]
        bboxes_corners = deepcopy(results['gt_bboxes_3d']._data.corners) #[num,8,3]
        
        # 对 gt_bboxes.corners 进行处理（增加随机偏移和随机丢弃）
        bboxes_with_offset, bboxes_corners_with_offset = self.random_offset_bboxes_x(bboxes, bboxes_corners, self.change_prob)
        new_bboxes, new_bboxes_corners = self.random_drop_bboxes(bboxes_with_offset, bboxes_corners_with_offset, self.drop_prob)
     
        #print("new_bboxes: ", new_bboxes)
        
        # 将新的 new_bboxes 添加到 results 中
        results['objectlist_bbox'] = DC(new_bboxes, cpu_only=False)
        results['objectlist_bbox_corner'] = DC(new_bboxes_corners, cpu_only=False)
        return results


    def random_offset_bboxes_x(self, bboxes, boxes_corners, change_prob):
        num, _  = bboxes.size()
        
        # 随机选择一些物体
        indices = torch.randperm(num)[:int(num * change_prob)]        
        # 随机生成x轴的偏移量
        x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1))        
        # 只对选中物体的x轴应用偏移
        bboxes[indices, 0] += x_offsets.squeeze()
        
        # 随机生成x轴的偏移量
        #x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1, 1))
        x_offsets_corner = x_offsets.unsqueeze(1).expand(-1, 8, 1)        
        
        # 只对选中物体的x轴应用偏移
        boxes_corners[indices, :, 0] += x_offsets_corner.squeeze()

        return bboxes, boxes_corners
    

    def random_drop_bboxes(self, bboxes, boxes_corners, drop_prob):
        num_gt = bboxes.size(0)
        num_drop = int(num_gt * drop_prob)
    
        # 随机选择一些边界框
        drop_indices = torch.randperm(num_gt)[:num_drop]

        # 将这些边界框的所有元素设置为0
        for index in drop_indices:
            bboxes[index] = 0
            boxes_corners[index] = 0
        
        bboxes_without0 = bboxes[bboxes[:, 3] != 0]
        boxes_corners_without0 = boxes_corners[(boxes_corners[:, 0, 0] != 0) & (boxes_corners[:, 0, 1] != 0) & (boxes_corners[:, 1, 0] != 0)]

        return bboxes_without0, boxes_corners_without0
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' offset_scale={self.offset_scale},'
        repr_str += f' change_prob={self.change_prob},'
        repr_str += f' drop_prob={self.drop_prob})'
        return repr_str

@PIPELINES.register_module()
class CreateObjectListWithNoise_forobjectlist_DQTrack(object):
    def __init__(self, with_object_list=True, offset_scale=2.0, change_prob=0.3, drop_prob=0.2):
        self.with_object_list = with_object_list
        self.offset_scale = offset_scale
        self.change_prob = change_prob
        self.drop_prob = drop_prob

    def __call__(self, results):
        gt_bboxes_3d = deepcopy(results['gt_bboxes_3d']._data)
        gt_labels_3d = deepcopy(results['gt_labels_3d']._data)
        objectlist_bbox = []
        objectlist_bbox_corner = []

        for num in range(len(gt_bboxes_3d)):
            # 提取 gt_bboxes
            bboxes1 = deepcopy(gt_bboxes_3d[num].gravity_center)        
            bboxes2 = deepcopy(gt_bboxes_3d[num].tensor[:,3:])#[num,6]
            bboxes_label = deepcopy(gt_labels_3d[num])#[num,]
            bboxes = torch.cat((bboxes1, bboxes2,bboxes_label.unsqueeze(1)), dim=1)#[num,10]
            bboxes_corners = deepcopy(gt_bboxes_3d[num].corners) #[num,8,3]

            # 对 gt_bboxes.corners 进行处理（增加随机偏移和随机丢弃）
            bboxes_with_offset, bboxes_corners_with_offset = self.random_offset_bboxes_x(bboxes, bboxes_corners, self.change_prob)
            new_bboxes, new_bboxes_corners = self.random_drop_bboxes(bboxes_with_offset, bboxes_corners_with_offset, self.drop_prob)

            objectlist_bbox.append(new_bboxes)
            objectlist_bbox_corner.append(new_bboxes_corners)

        results['objectlist_bbox'] = DC(objectlist_bbox, cpu_only=False)
        results['objectlist_bbox_corner'] = DC(objectlist_bbox_corner, cpu_only=False)
        return results


    def random_offset_bboxes_x(self, bboxes, boxes_corners, change_prob):
        num, _  = bboxes.size()
        
        # 随机选择一些物体
        indices = torch.randperm(num)[:int(num * change_prob)]        
        # 随机生成x轴的偏移量
        x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1))        
        # 只对选中物体的x轴应用偏移
        bboxes[indices, 0] += x_offsets.squeeze()
        
        # 随机生成x轴的偏移量
        #x_offsets = torch.randint(-int(offset_scale), int(offset_scale) + 1, (indices.size(0), 1, 1))
        x_offsets_corner = x_offsets.unsqueeze(1).expand(-1, 8, 1)        
        
        # 只对选中物体的x轴应用偏移
        boxes_corners[indices, :, 0] += x_offsets_corner.squeeze()

        return bboxes, boxes_corners
    

    def random_drop_bboxes(self, bboxes, boxes_corners, drop_prob):
        num_gt = bboxes.size(0)
        num_drop = int(num_gt * drop_prob)
    
        # 随机选择一些边界框
        drop_indices = torch.randperm(num_gt)[:num_drop]

        # 将这些边界框的所有元素设置为0
        for index in drop_indices:
            bboxes[index] = 0
            boxes_corners[index] = 0
        
        bboxes_without0 = bboxes[bboxes[:, 3] != 0]
        boxes_corners_without0 = boxes_corners[(boxes_corners[:, 0, 0] != 0) & (boxes_corners[:, 0, 1] != 0) & (boxes_corners[:, 1, 0] != 0)]

        return bboxes_without0, boxes_corners_without0
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' offset_scale={self.offset_scale},'
        repr_str += f' change_prob={self.change_prob},'
        repr_str += f' drop_prob={self.drop_prob})'
        return repr_str
    
@PIPELINES.register_module()
class CreatePseudoObjectList(object):
    def __init__(self, with_object_list=True, max_noise_std=0.1, max_drop_rate=0.1, max_fp_rate=0.05, max_split_rate=0.3, seed=42):
        self.with_object_list = with_object_list
        self.max_noise_std = max_noise_std
        self.max_fp_rate = max_fp_rate
        self.max_drop_rate = max_drop_rate
        self.max_split_rate = max_split_rate
        self.seed = seed
    
    def generate_random_box(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        translation = np.random.uniform([-51.2, -51.2, -5.0], [51.2, 51.2, 3.0])  # Random translation within a reasonable range
        size = np.random.uniform(0.5, 5, 3)            # Random size
        yaw = np.random.uniform(-np.pi, np.pi, 1)    # Random yaw angle
        velocity = np.random.uniform(-10, 10, 2)       # Random velocity
        label = np.random.randint(0, 11, 1)          # Random label between 0 and 10
        return np.concatenate([translation, size, yaw, velocity, label])
    
    def corners(self, pseudo_bboxes):
        # Convert pseudo_bboxes to LiDARInstance3DBoxes
        pseudo_boxes_instance = LiDARInstance3DBoxes(pseudo_bboxes[:, :7])

        # Get corners in clockwise order
        corners = pseudo_boxes_instance.corners

        return corners
    
    def __call__(self, results):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        # Randomize noise and rates within the specified maximum values
        current_noise_std = np.random.uniform(0, self.max_noise_std)
        current_drop_rate = np.random.uniform(0, self.max_drop_rate)
        current_false_positive_rate = np.random.uniform(0, self.max_fp_rate)
        current_split_rate = np.random.uniform(0, self.max_split_rate)

        # Extract ground truth bboxes
        gt_bboxes_3d = deepcopy(results['gt_bboxes_3d']._data)
        gt_labels_3d = deepcopy(results['gt_labels_3d']._data)
        objectlist_bbox_gt_match = []
        objectlist_bbox = []
        objectlist_bbox_corner = []
        for num in range(len(gt_bboxes_3d)):
            # 提取 gt_bboxes
            bboxes1 = deepcopy(gt_bboxes_3d[num].gravity_center)        
            bboxes2 = deepcopy(gt_bboxes_3d[num].tensor[:,3:])#[num,6]
            bboxes_label = deepcopy(gt_labels_3d[num])#[num,]
            bboxes = torch.cat((bboxes1, bboxes2,bboxes_label.unsqueeze(1)), dim=1)#[num,10]

            num_bboxes = bboxes.shape[0]

            # Calculate noise standard deviation for each axis proportional to the absolute coordinates
            noise_std = current_noise_std * bboxes[:, :3].abs()  # [num, 3]

            # Apply different noise levels to x, y, and z coordinates
            noise = torch.randn_like(bboxes[:, :3]) * noise_std

            # Apply noise to bboxes
            noisy_bboxes = bboxes.clone()
            noisy_bboxes[:, :3] += noise  # Apply noise to x, y, z coordinates

            # Introduce noise to sizes (size_x, size_y, size_z)
            size_noise = torch.randn_like(bboxes[:, 3:6]) * current_noise_std
            noisy_bboxes[:, 3:6] += size_noise  # Apply noise to size_x, size_y, size_z

            # Introduce noise to velocities (v_x, v_y)
            velocity_noise = torch.randn_like(bboxes[:, 8:10]) * current_noise_std
            noisy_bboxes[:, 8:10] += velocity_noise  # Apply noise to v_x, v_y
            objectlist_bbox_gt_match.append(noisy_bboxes)

            # Drop some true positives
            keep_mask = torch.rand(num_bboxes) > current_drop_rate
            noisy_bboxes = noisy_bboxes[keep_mask]

            # Add false positives
            num_false_positives = int(current_false_positive_rate * num_bboxes)
            false_positive_bboxes = []
            for _ in range(num_false_positives):
                false_positive_bbox = self.generate_random_box()
                false_positive_bboxes.append(false_positive_bbox)

            # Convert false positives to torch tensors and concatenate with noisy bboxes
            if false_positive_bboxes:
                false_positive_bboxes = torch.tensor(false_positive_bboxes, dtype=torch.float32)
                pseudo_bboxes = torch.cat((noisy_bboxes, false_positive_bboxes), dim=0)
            else:
                pseudo_bboxes = noisy_bboxes

            # Add noise to labels
            noisy_labels = pseudo_bboxes[:, -1].clone()
            for i in range(noisy_labels.size(0)):
                if torch.rand(1).item() < current_split_rate:
                    if noisy_labels[i] < 6:
                        noisy_labels[i] = np.random.choice([1, 2, 3, 4, 5, 10])
                    else:
                        noisy_labels[i] = np.random.choice([6, 7, 8, 9, 10])

            # Put noisy_labels back into pseudo_bboxes
            pseudo_bboxes[:, -1] = noisy_labels.long()

            # Shuffle the pseudo object list
            indices = torch.randperm(pseudo_bboxes.size(0))
            pseudo_bboxes = pseudo_bboxes[indices]

            # Get corners of pseudo_bboxes
            pseudo_bboxes_corners = self.corners(pseudo_bboxes)

            objectlist_bbox.append(pseudo_bboxes)
            objectlist_bbox_corner.append(pseudo_bboxes_corners)

        # Update results
        if self.with_object_list:
            results['objectlist_bbox'] = DC(objectlist_bbox_gt_match, cpu_only=False)
        else:
            results['objectlist_bbox'] = DC(objectlist_bbox, cpu_only=False)
        results['objectlist_bbox_corner'] = DC(objectlist_bbox_corner, cpu_only=False)

        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' max_noise_std={self.max_noise_std},'
        repr_str += f' max_fp_rate={self.max_fp_rate},'
        repr_str += f' max_drop_rate={self.max_drop_rate})'
        repr_str += f' max_split_rate={self.max_split_rate})'
        repr_str += f' seed={self.seed})'
        return repr_str 