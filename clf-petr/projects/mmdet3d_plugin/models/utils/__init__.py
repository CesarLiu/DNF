# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .dgcnn_attn import DGCNNAttn
from .detr import Deformable3DDetrTransformerDecoder
from .detr3d_transformer import Detr3DTransformer, Detr3DTransformerDecoder, Detr3DCrossAtten
from .positional_encoding import SinePositionalEncoding3D, LearnedPositionalEncoding3D
from .petr_transformer import PETRTransformer, PETRDNTransformer, PETRMultiheadAttention, PETRTransformerEncoder, PETRTransformerDecoder
from .petr_transformer_version1 import PETRTransformer_version1, PETRMultiheadAttention_version1, PETRTransformerEncoder_version1, PETRTransformerDecoder_version1, SMCAMultiheadAttention,PETRGaussianMultiheadAttention_1
from .petr_transformer_version2 import PETRTransformer_version2, PETRMultiheadAttention_version2, PETRTransformerEncoder_version2, PETRTransformerDecoder_version2
from .petr_transformer_version3 import PETRTransformer_version3, PETRDNTransformer_GMHA, PETRTransformerEncoder_version3, PETRTransformerDecoder_GMHA, PETRTransformerDecoderLayer_GMHA, SMCAMultiheadAttention_verison3, PETRGaussianMultiheadAttention

__all__ = ['DGCNNAttn', 'Deformable3DDetrTransformerDecoder', 
           'Detr3DTransformer', 'Detr3DTransformerDecoder', 'Detr3DCrossAtten'
           'SinePositionalEncoding3D', 'LearnedPositionalEncoding3D',
           'PETRTransformer', 'PETRDNTransformer', 'PETRMultiheadAttention', 
           'PETRTransformerEncoder', 'PETRTransformerDecoder',
           'PETRGaussianMultiheadAttention', 'GaussianMultiheadAttention',
              'PETRTransformer_version1', 'PETRMultiheadAttention_version1',
                'PETRTransformerEncoder_version1', 'PETRTransformerDecoder_version1',
                'SMCAMultiheadAttention','PETRGaussianMultiheadAttention_1',
                'PETRTransformer_version2', 'PETRMultiheadAttention_version2',
                'PETRTransformerEncoder_version2', 'PETRTransformerDecoder_version2',
                'PETRTransformer_version3', 'PETRDNTransformer_GMHA',
                'PETRTransformerEncoder_version3', 'PETRTransformerDecoder_GMHA',
                'PETRTransformerDecoderLayer_GMHA', 'SMCAMultiheadAttention_verison3',
                'PETRGaussianMultiheadAttention',
           ]


