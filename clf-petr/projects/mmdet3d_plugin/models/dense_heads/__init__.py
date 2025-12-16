# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from .dgcnn3d_head import DGCNN3DHead
from .detr3d_head import Detr3DHead
from .petr_head import PETRHead
from .petr_dnhead import PETRDNHead
from .petrv2_head import PETRv2Head
from .petr_head_seg import PETRHead_seg
from .petrv2_dnhead import PETRv2DNHead
from .petrv2_head_version1 import PETRv2Head_version1
from .petrv2_head_version2 import PETRv2Head_version2
from .petrv2_head_version3 import PETRv2Head_version3
from .petrv2_head_version4 import PETRv2Head_version4
from .petrv2_head_version5 import PETRv2Head_version5
from .petrv2_dnhead_version1 import PETRv2DNHead_version1
from .petrv2_dnhead_version1_test import PETRv2DNHead_version1_test
from .petrv2_dnhead_gm import PETRv2DNHeadGM
from .petrv2_dnhead_version2_testwithoutobj import PETRv2DNHeadGM_testwithout
__all__ = ['DGCNN3DHead', 'Detr3DHead','PETRHead','PETRv2Head','PETRHead_seg', 'PETRv2DNHead','PETRv2Head_version1',
           'PETRv2Head_version2','PETRv2Head_version3','PETRv2Head_version4','PETRv2Head_version5','PETRv2DNHead_version1',
           'PETRv2DNHead_version1_test','PETRv2DNHeadGM','PETRv2DNHeadGM_testwithout']