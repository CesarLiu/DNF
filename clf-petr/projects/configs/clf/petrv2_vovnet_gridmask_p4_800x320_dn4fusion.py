_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP', ###use checkpoint to save memory
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='PETRv2DNHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        dn_for_fusion=0,
        scalar = 5, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRDNTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'gt_bboxes_3d','gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='LoadAnnotations3D_Test', with_bbox_3d=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            # dict(type='CreateObjectListWithNoise_fordn',with_object_list=True, offset_scale=2.0, change_prob=0.3, drop_prob=0.2),
            dict(type='CreatePseudoObjectList',with_object_list=True, max_noise_std=0.06, max_drop_rate=0.2, max_fp_rate=0.1, max_split_rate=0.3,seed=0),
            dict(type='Collect3D', keys=['img','objectlist_bbox'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_train.pkl',#'mmdet3d_nuscenes_petr_mini_train.pkl'
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl', classes=class_names, modality=input_modality))


optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )
total_epochs = 24
evaluation = dict(interval=4, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=4, max_keep_ckpts=3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from=None

# Evaluating bboxes of pts_bbox
# mAP: 0.4191
# mATE: 0.7005
# mASE: 0.2632
# mAOE: 0.4555
# mAVE: 0.3814
# mAAE: 0.1878
# NDS: 0.5107
# Eval time: 235.6s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.601   0.496   0.148   0.072   0.321   0.187
# truck   0.388   0.724   0.200   0.106   0.321   0.206
# bus     0.447   0.756   0.189   0.096   0.721   0.229
# trailer 0.235   1.047   0.233   0.569   0.340   0.138
# construction_vehicle    0.135   0.965   0.464   1.150   0.154   0.344
# pedestrian      0.490   0.659   0.289   0.531   0.424   0.186
# motorcycle      0.414   0.634   0.255   0.524   0.505   0.197
# bicycle 0.405   0.578   0.257   0.908   0.264   0.015
# traffic_cone    0.569   0.526   0.316   nan     nan     nan
# barrier 0.507   0.619   0.281   0.144   nan     nan

#### mixed attn mask
# mAP: 0.3738                                                                                                                                                                                               
# mATE: 0.7639
# mASE: 0.2777
# mAOE: 0.4588
# mAVE: 0.4427
# mAAE: 0.1922
# NDS: 0.4734
# Eval time: 148.4s

# Per-class results:
# Object Class            AP      ATE     ASE     AOE     AVE     AAE   
# car                     0.522   0.561   0.158   0.084   0.403   0.194 
# truck                   0.340   0.757   0.216   0.101   0.383   0.205 
# bus                     0.426   0.764   0.205   0.122   0.817   0.256 
# trailer                 0.214   1.126   0.252   0.577   0.420   0.143 
# construction_vehicle    0.116   1.097   0.499   1.143   0.147   0.350 
# pedestrian              0.442   0.700   0.291   0.543   0.444   0.190 
# motorcycle              0.357   0.701   0.270   0.592   0.612   0.186 
# bicycle                 0.344   0.678   0.263   0.785   0.315   0.013 
# traffic_cone            0.513   0.571   0.327   nan     nan     nan   
# barrier                 0.464   0.685   0.297   0.182   nan     nan
#### mixed attn mask without object list in inference
# mAP: 0.4094                                                                                                                                                                              
# mATE: 0.7379
# mASE: 0.2661
# mAOE: 0.4572
# mAVE: 0.4211
# mAAE: 0.1887
# NDS: 0.4976
# Eval time: 122.5s

# Per-class results:
# Object Class            AP      ATE     ASE     AOE     AVE     AAE   
# car                     0.586   0.512   0.148   0.082   0.372   0.189 
# truck                   0.371   0.726   0.201   0.097   0.370   0.202 
# bus                     0.444   0.734   0.198   0.121   0.822   0.261 
# trailer                 0.232   1.138   0.232   0.570   0.397   0.139 
# construction_vehicle    0.133   1.071   0.477   1.156   0.135   0.339 
# pedestrian              0.485   0.680   0.286   0.541   0.439   0.192 
# motorcycle              0.389   0.692   0.257   0.591   0.536   0.176 
# bicycle                 0.377   0.650   0.254   0.776   0.298   0.012 
# traffic_cone            0.570   0.529   0.319   nan     nan     nan   
# barrier                 0.506   0.647   0.289   0.181   nan     nan