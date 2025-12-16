# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import mmcv
import torch
import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import LiDARInstance3DBoxes, BaseInstance3DBoxes
from mmdet3d.datasets.pipelines import DefaultFormatBundle
import random
from mmcv.parallel import DataContainer as DC
from copy import deepcopy


@PIPELINES.register_module()
class CreateObjectListWithNoise(object):
    def __init__(self, with_object_list=True, offset_scale=2.0, change_prob=0.3, drop_prob=0.2):
        self.with_object_list = with_object_list
        self.offset_scale = offset_scale
        self.change_prob = change_prob
        self.drop_prob = drop_prob

    def __call__(self, results):
        # 提取 gt_bboxes
        #bboxes = deepcopy(results['gt_bboxes_3d']._data.tensor)#[num,9]
        bboxes = deepcopy(results['gt_bboxes_3d']._data.corners) #[num,8,3]
        # 对 gt_bboxes.corners 进行处理（增加随机偏移和随机丢弃）
        bboxes_with_offset = self.random_offset_bboxes_x(bboxes, self.change_prob)
        new_bboxes = self.random_drop_bboxes(bboxes_with_offset, self.drop_prob)
        #print("new_bboxes: ", new_bboxes)
        
        # 将新的 new_bboxes 添加到 results 中
        results['objectlist_bbox'] = DC(new_bboxes, cpu_only=False)
        return results


    def random_offset_bboxes_x(self, bboxes, change_prob):
        num, _, _ = bboxes.size()
        
        # 随机选择一些物体
        indices = torch.randperm(num)[:int(num * change_prob)]
        
        # 随机生成x轴的偏移量
        x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1, 1))
        
        # 扩展偏移量到所有角点的x坐标
        x_offsets = x_offsets.expand(-1, 8, 1)
        
        # 只对选中物体的x轴应用偏移
        bboxes[indices, :, 0] += x_offsets.squeeze()
        
        return bboxes
    

    def random_drop_bboxes(self, bboxes, drop_prob):
        num_gt = bboxes.size(0)
        num_drop = int(num_gt * drop_prob)

        # 随机选择一些边界框
        drop_indices = torch.randperm(num_gt)[:num_drop]

        # 将这些边界框的所有元素设置为0
        for index in drop_indices:
            bboxes[index] = 0
        return bboxes
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' offset_scale={self.offset_scale},'
        repr_str += f' change_prob={self.change_prob},'
        repr_str += f' drop_prob={self.drop_prob})'
        return repr_str
    

@PIPELINES.register_module()
class CreateObjectListWithNoise_withModalMask3D(object):
    def __init__(self, mode='test', with_object_list=True, offset_scale=5.0, change_prob=0.3, drop_prob=0.2):
        self.mode = mode
        self.with_object_list = with_object_list
        self.offset_scale = offset_scale
        self.change_prob = change_prob
        self.drop_prob = drop_prob

    def __call__(self, results):
        # 提取 gt_bboxes
        #bboxes = deepcopy(results['gt_bboxes_3d']._data.tensor)#[num,9]
        bboxes = deepcopy(results['gt_bboxes_3d']._data.corners) #[num,8,3]
        if self.mode == 'train':
            seed = np.random.rand()            
            if seed > 0.5:
                new_bboxes = np.abs(bboxes * 0) + 1e-5
            else:
                # 对 gt_bboxes.corners 进行处理（增加随机偏移和随机丢弃）
                bboxes_with_offset = self.random_offset_bboxes_x(bboxes, self.change_prob)
                new_bboxes = self.random_drop_bboxes(bboxes_with_offset, self.drop_prob)
                #print("new_bboxes: ", new_bboxes)
        
        # 将新的 new_bboxes 添加到 results 中
        results['objectlist_bbox'] = DC(new_bboxes, cpu_only=False)
        return results


    def random_offset_bboxes_x(self, bboxes, change_prob):
        num, _, _ = bboxes.size()
        
        # 随机选择一些物体
        indices = torch.randperm(num)[:int(num * change_prob)]
        
        # 随机生成x轴的偏移量
        x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1, 1))
        
        # 扩展偏移量到所有角点的x坐标
        x_offsets = x_offsets.expand(-1, 8, 1)
        
        # 只对选中物体的x轴应用偏移
        bboxes[indices, :, 0] += x_offsets.squeeze()
        
        return bboxes
    

    def random_drop_bboxes(self, bboxes, drop_prob):
        num_gt = bboxes.size(0)
        num_drop = int(num_gt * drop_prob)

        # 随机选择一些边界框
        drop_indices = torch.randperm(num_gt)[:num_drop]

        # 将这些边界框的所有元素设置为0
        for index in drop_indices:
            bboxes[index] = 0
        return bboxes
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' offset_scale={self.offset_scale},'
        repr_str += f' change_prob={self.change_prob},'
        repr_str += f' drop_prob={self.drop_prob})'
        return repr_str
    

   
@PIPELINES.register_module()
class CreateObjectListWithNoise_fordn(object):
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
        #bboxes = deepcopy(results['gt_bboxes_3d']._data.corners) #[num,8,3]
        
        # 对 gt_bboxes.corners 进行处理（增加随机偏移和随机丢弃）
        bboxes_with_offset = self.random_offset_bboxes_x(bboxes, self.change_prob)
        new_bboxes = self.random_drop_bboxes(bboxes_with_offset, self.drop_prob)
     
        #print("new_bboxes: ", new_bboxes)
        
        # 将新的 new_bboxes 添加到 results 中
        results['objectlist_bbox'] = DC(new_bboxes, cpu_only=False)
        return results


    def random_offset_bboxes_x(self, bboxes, change_prob):
        num, _  = bboxes.size()
        
        # 随机选择一些物体
        indices = torch.randperm(num)[:int(num * change_prob)]        
        # 随机生成x轴的偏移量
        x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1))        
        # 只对选中物体的x轴应用偏移
        bboxes[indices, 0] += x_offsets.squeeze()
        
        return bboxes
    

    def random_drop_bboxes(self, bboxes, drop_prob):
        num_gt = bboxes.size(0)
        num_drop = int(num_gt * drop_prob)
    
        # 随机选择一些边界框
        drop_indices = torch.randperm(num_gt)[:num_drop]

        # 将这些边界框的所有元素设置为0
        for index in drop_indices:
            bboxes[index] = 0
        
        bboxes_without0 = bboxes[bboxes[:, 3] != 0]
        return bboxes_without0
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' offset_scale={self.offset_scale},'
        repr_str += f' change_prob={self.change_prob},'
        repr_str += f' drop_prob={self.drop_prob})'
        return repr_str
    

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
class CreatePseudoObjectList(object):
    def __init__(self, training=True, max_noise_std=0.1, max_drop_rate=0.1, max_fp_rate=0.05, max_split_rate=0.3, seed=42):
        self.training = training
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
        bboxes1 = deepcopy(results['gt_bboxes_3d']._data.gravity_center)  # [num,3] - x, y, z
        bboxes2 = deepcopy(results['gt_bboxes_3d']._data.tensor[:, 3:])  # [num,6] - size_x, size_y, size_z, yaw, v_x, v_y
        bboxes_label = deepcopy(results['gt_labels_3d']._data)  # [num,] - label
        bboxes = torch.cat((bboxes1, bboxes2, bboxes_label.unsqueeze(1)), dim=1)  # [num,10]

        # num_bboxes = bboxes.shape[0]

        # # Calculate noise standard deviation for each axis proportional to the absolute coordinates
        # noise_std = current_noise_std * bboxes[:, :3].abs()  # [num, 3]

        # # Apply different noise levels to x, y, and z coordinates
        # noise = torch.randn_like(bboxes[:, :3]) * noise_std

        # # Apply noise to bboxes
        # noisy_bboxes = bboxes.clone()
        # noisy_bboxes[:, :3] += noise  # Apply noise to x, y, z coordinates

        # # Introduce noise to sizes (size_x, size_y, size_z)
        # size_noise = torch.randn_like(bboxes[:, 3:6]) * current_noise_std
        # noisy_bboxes[:, 3:6] += size_noise  # Apply noise to size_x, size_y, size_z

        # # Introduce noise to velocities (v_x, v_y)
        # velocity_noise = torch.randn_like(bboxes[:, 8:10]) * current_noise_std
        # noisy_bboxes[:, 8:10] += velocity_noise  # Apply noise to v_x, v_y
        # results['objectlist_bbox'] = DC(noisy_bboxes, cpu_only=False)

        # # Drop some true positives
        # keep_mask = torch.rand(num_bboxes) > current_drop_rate
        # noisy_bboxes = noisy_bboxes[keep_mask]

        # # Add false positives
        # num_false_positives = int(current_false_positive_rate * num_bboxes)
        # false_positive_bboxes = []
        # for _ in range(num_false_positives):
        #     false_positive_bbox = self.generate_random_box()
        #     false_positive_bboxes.append(false_positive_bbox)

        # # Convert false positives to torch tensors and concatenate with noisy bboxes
        # if false_positive_bboxes:
        #     false_positive_bboxes = torch.tensor(false_positive_bboxes, dtype=torch.float32)
        #     pseudo_bboxes = torch.cat((noisy_bboxes, false_positive_bboxes), dim=0)
        # else:
        #     pseudo_bboxes = noisy_bboxes

        # # Add noise to labels
        # noisy_labels = pseudo_bboxes[:, -1].clone()
        # for i in range(noisy_labels.size(0)):
        #     if torch.rand(1).item() < current_split_rate:
        #         if noisy_labels[i] < 6:
        #             noisy_labels[i] = np.random.choice([1, 2, 3, 4, 5, 10])
        #         else:
        #             noisy_labels[i] = np.random.choice([6, 7, 8, 9, 10])

        # # Put noisy_labels back into pseudo_bboxes
        # pseudo_bboxes[:, -1] = noisy_labels.long()

        # # Shuffle the pseudo object list
        # indices = torch.randperm(pseudo_bboxes.size(0))
        # pseudo_bboxes = pseudo_bboxes[indices]

        # Get corners of pseudo_bboxes
        # pseudo_bboxes_corners = self.corners(pseudo_bboxes)
        pseudo_bboxes_corners = self.corners(bboxes)


        # Update results
        if not self.training:
            # results['objectlist_bbox'] = DC(pseudo_bboxes, cpu_only=False)
            results['objectlist_bbox'] = DC(bboxes, cpu_only=False)
        results['objectlist_bbox_corner'] = DC(pseudo_bboxes_corners, cpu_only=False)

        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(training={self.training},'
        repr_str += f' max_noise_std={self.max_noise_std},'
        repr_str += f' max_fp_rate={self.max_fp_rate},'
        repr_str += f' max_drop_rate={self.max_drop_rate})'
        repr_str += f' max_split_rate={self.max_split_rate})'
        repr_str += f' seed={self.seed})'
        return repr_str  
""" @PIPELINES.register_module()
class CreateObjectListWithNoise_forvis(object):
    def __init__(self, with_object_list=True, offset_scale=5.0, change_prob=0.3, drop_prob=0.2):
        self.with_object_list = with_object_list
        self.offset_scale = offset_scale
        self.change_prob = change_prob
        self.drop_prob = drop_prob

    def __call__(self, results):
        # 提取 gt_bboxes
        #bboxes = deepcopy(results['gt_bboxes_3d']._data.tensor)#[num,9]
        #new_bboxes = deepcopy(results['gt_bboxes_3d']._data.corners) #[num,8,3]
        new_bboxes = deepcopy(results['gt_bboxes_3d']._data.corners)
        # 对 gt_bboxes.corners 进行处理（增加随机偏移和随机丢弃）
        bboxes_with_offset = self.random_offset_bboxes_x(new_bboxes, self.change_prob)
        new_bboxes = self.random_drop_bboxes(bboxes_with_offset, self.drop_prob)
        #print("new_bboxes: ", new_bboxes)
        
        # 将新的 new_bboxes 添加到 results 中
        results['objectlist_bbox'] = DC(new_bboxes, cpu_only=False)
        return results


    # def random_offset_bboxes(self, bboxes, change_prob):
    #     num_gt, num_points, num_coords = bboxes.size()
    
    #     # 随机选择一些对象
    #     indices = torch.randperm(num_gt)[:int(num_gt * change_prob)]        
    #     # 随机生成偏移量
    #     offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), num_points, num_coords))        
    #     # 对选中的对象的xyz坐标应用偏移
    #     bboxes[indices] += offsets
    #     return bboxes
    def random_offset_bboxes_x(self, bboxes, change_prob):
        num, _, _ = bboxes.size()
        
        # 随机选择一些物体
        indices = torch.randperm(num)[:int(num * change_prob)]
        
        # 随机生成x轴的偏移量
        x_offsets = torch.randint(-int(self.offset_scale), int(self.offset_scale) + 1, (indices.size(0), 1, 1))
        
        # 扩展偏移量到所有角点的x坐标
        x_offsets = x_offsets.expand(-1, 8, 1)
        
        # 只对选中物体的x轴应用偏移
        bboxes[indices, :, 0] += x_offsets.squeeze()
        
        return bboxes
    

    def random_drop_bboxes(self, bboxes, drop_prob):
        num_gt = bboxes.size(0)
        num_drop = int(num_gt * drop_prob)

        # 随机选择一些边界框
        drop_indices = torch.randperm(num_gt)[:num_drop]

        # 将这些边界框的所有元素设置为0
        for index in drop_indices:
            bboxes[index] = 0
        return bboxes
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_object_list={self.with_object_list},'
        repr_str += f' offset_scale={self.offset_scale},'
        repr_str += f' change_prob={self.change_prob},'
        repr_str += f' drop_prob={self.drop_prob})'
        return repr_str
 """