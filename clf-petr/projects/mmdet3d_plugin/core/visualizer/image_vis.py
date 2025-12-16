# Copyright (c) OpenMMLab. All rights reserved.
import copy

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def _plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        # filter boxes out of range
        h,w,c = img.shape
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int)
            for idx, corner in enumerate(corners):
                corners[idx][0] = w if corner[0] > w else corner[0]
                corners[idx][0] = 0 if corner[0] < 0 else corner[0]
                corners[idx][1] = w if corner[1] > h else corner[1]
                corners[idx][1] = 0 if corner[1] < 0 else corner[1]
            # draw
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)

        return img.astype(np.uint8)

def _draw_lidar_bbox3d(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
        """Project the 3D bbox on 2D plane and draw on input image.

        Args:
            bboxes3d (:obj:`LiDARInstance3DBoxes`):
                3d bbox in lidar coordinate system to visualize.
            raw_img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            img_metas (dict): Useless here.
            color (tuple[int], optional): The color to draw bboxes.
                Default: (0, 255, 0).
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        
        img = raw_img.copy()
        corners_3d = bboxes3d.cpu().numpy()
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
                        [corners_3d.reshape(-1, 3),
                        np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

        return _plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

