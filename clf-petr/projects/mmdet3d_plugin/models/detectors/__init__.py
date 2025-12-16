# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from .obj_dgcnn import ObjDGCNN
from .detr3d import Detr3D
from .petr3d import Petr3D
from .petr3d_seg import Petr3D_seg
from .mspetr3d import MSPetr3D
from .petr3d_version1 import Petr3D_version1
from .petr3d_version2 import Petr3D_version2
from .petr3d_version3 import Petr3D_version3
from .petr3d_version4 import Petr3D_version4
from .petr3d_version5 import Petr3D_version5
from .petr3d_fordn import Petr3D_fordn
from .petr3d_fordn_test import Petr3D_fordn_test
from .Petr3DCLF import Petr3DCLF
from .petr3d_objectlist_testwithout import Petr3D_objectlist_testwithout
__all__ = ['ObjDGCNN', 'Detr3D', 'Petr3D', 'MSPetr3D', 'Petr3D_seg','Petr3D_version1',
           'Petr3D_version2','Petr3D_version3','Petr3D_version4','Petr3D_version5','Petr3D_fordn', 'Petr3D_fordn_test','Petr3DCLF','Petr3D_objectlist_testwithout']
