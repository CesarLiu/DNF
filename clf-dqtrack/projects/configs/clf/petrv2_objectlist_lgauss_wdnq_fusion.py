###
# with denoising with objectlist + gaussina mask
###
_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
bev_stride = 4
track_frame = 3
fp16_enabled = True

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

track_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

velocity_error = {
  'car':6,
  'truck':6,
  'bus':6,
  'trailer':5,
  'pedestrian':5,
  'motorcycle':12,
  'bicycle':5,  
}
velocity_error = [velocity_error[_name] for _name in track_names]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='PETRTrackerDQ_objectlist_dnq_static',
    # not use grid mask
    use_grid_mask=False,
    tracker_cfg=dict(
        track_frame=track_frame,
        num_track=300,
        num_query=900,
        num_cams=6,
        embed_dims=256,
        ema_decay=0.5,
        scalar = 10, ##noise groups
        train_det_only=False,
        prop_dnq=False,
        train_track_only=True,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names,
        ),
        pos_cfg=dict(
            pos_trans='ffn',
            fuse_type="sum",
            final_trans="linear",
        ),
        query_trans=dict(
            with_att=True,
            with_pos=True,
            min_channels=256,
            drop_rate=0.0,
        ),
        track_aug=dict(
            drop_prob=0,
            fp_ratio=0.2,
        ),
        # used for training
        ema_drop=0.0,
        # used for inference
        class_spec = True,
        eval_det_only=False,        
        velo_error=velocity_error, 
        assign_method='hungarian',
        det_thres=0.3,
        new_born_thres=0.2,
        asso_thres=0.1,
        miss_thres=7,
    ),
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4', 'stage5')),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='PETRv2TrackDNHead_objectlist_dnq_fusion',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        scalar = 10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRDNTransformer_objectlist',
            decoder=dict(
                type='PETRTransformerDecoder_objectlist',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer_objectlist',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRGaussianMultiheadAttention_objectlist',
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
            type='SinePositionalEncoding3D', 
            num_feats=128, 
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=100), 
    loss_cfg=dict(
        type='DualQueryMatcher',
        num_classes=10,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost.
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_asso=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0),
        loss_me=dict(
            type="CategoricalCrossEntropyLoss",
            loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=bev_stride,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'NuScenesDatasetPETRTrack'
data_root = 'data/nuscenes/'
info_root = 'data/infos/'
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
    dict(type='TrackletRangeFilter', point_cloud_range=point_cloud_range),
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
]
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='CreatePseudoObjectList',with_object_list=True, max_noise_std=0.06, max_drop_rate=0.2, max_fp_rate=0.1, max_split_rate=0.3,seed=0),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'lidar_timestamp','objectlist_bbox', 'objectlist_bbox_corner'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='LoadAnnotations3D_Test', with_bbox_3d=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    # dict(type='CreatePseudoObjectList',with_object_list=False, max_noise_std=0.03, max_drop_rate=0.1, max_fp_rate=0.05, max_split_rate=0.1,seed=0),# small
    # dict(type='CreatePseudoObjectList',with_object_list=True, max_noise_std=0.015, max_drop_rate=0.05, max_fp_rate=0.03, max_split_rate=0.05,seed=0),# smaller
    dict(type='CreatePseudoObjectList',with_object_list=False, max_noise_std=0.06, max_drop_rate=0.2, max_fp_rate=0.1, max_split_rate=0.3,seed=0),
    # dict(type='CreatePseudoObjectList',with_object_list=False, max_noise_std=0.1, max_drop_rate=0.3, max_fp_rate=0.2, max_split_rate=0.5,seed=0),
    dict(type='CollectUnified3D', keys=['img', 'lidar_timestamp','objectlist_bbox', 'objectlist_bbox_corner'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=info_root + 'mmdet3d_nuscenes_30f_infos_train.pkl',
        num_frames_per_sample=track_frame,  # number of frames for each 
        pipeline=train_pipeline,
        pipeline_post=train_pipeline_post,
        classes=class_names,
        track_classes=track_names,
        modality=input_modality,
        test_mode=False,
        force_continuous=True, # force to use continuous frame
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, pipeline_post=test_pipeline_post, 
             classes=class_names, track_classes=track_names, modality=input_modality, 
             ann_file=info_root + 'mmdet3d_nuscenes_30f_infos_val.pkl', 
             num_frames_per_sample=1), # please change to your own info file
    test=dict(type=dataset_type, pipeline=test_pipeline, pipeline_post=test_pipeline_post, 
              classes=class_names, track_classes=track_names, modality=input_modality,
              ann_file=info_root + 'mmdet3d_nuscenes_30f_infos_val.pkl',
              num_frames_per_sample=1))
    
optimizer = dict(
    type='AdamW', 
    lr=1e-5,#1e-5
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=105, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 6
evaluation = dict(interval=2, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=6)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/epoch_24.pth' 
# resume_from='work_dirs/guo_dn_as_fusion_no_attn_w_lgauss_wdnq_fusion/epoch_3.pth'
fp16 = dict(loss_scale=32.)

# pts_bbox_NuScenes/amota: 0.4679, pts_bbox_NuScenes/amotp: 1.2660

# Evaluating bboxes of pts_bbox
# mAP: 0.6557                                                                                                                                          
# mATE: 0.4549
# mASE: 0.2296
# mAOE: 0.5422
# mAVE: 0.4848
# mAAE: 0.1924
# NDS: 0.6375
# Eval time: 101.1s

# pts_bbox_NuScenes/amota: 0.5884, pts_bbox_NuScenes/amotp: 1.0194

# Per-class results:
#                 AMOTA   AMOTP   RECALL  MOTAR   GT      MOTA    MOTP    MT      ML      FAF     TP      FP      FN      IDS     FRAG    TID     LGD
# bicycle         0.570   1.035   0.611   0.852   1993    0.515   0.475   61      42      12.8    1205    178     776     12      14      0.81    1.28
# bus             0.582   1.244   0.705   0.717   2112    0.498   0.914   48      17      26.3    1465    414     624     23      85      0.80    1.91
# car             0.794   0.746   0.834   0.845   58317   0.686   0.598   2558    452     128.0   47380   7358    9689    1248    1284    0.29    0.76
# motorcy         0.514   1.061   0.562   0.826   1977    0.457   0.526   41      46      14.1    1093    190     865     19      26      1.60    2.31
# pedestr         0.811   0.678   0.874   0.891   25423   0.750   0.538   1300    105     54.6    21421   2344    3208    794     660     0.28    0.66
# trailer         0.113   1.618   0.207   0.553   2425    0.113   0.974   13      106     22.6    497     222     1923    5       36      1.74    3.03
# truck           0.502   1.106   0.566   0.660   9650    0.367   0.754   198     194     48.9    5360    1823    4188    102     225     0.73    1.66

# Aggregated results:
# AMOTA   0.555
# AMOTP   1.070
# RECALL  0.623
# MOTAR   0.763
# GT      14556
# MOTA    0.484
# MOTP    0.683
# MT      4219
# ML      962
# FAF     43.9
# TP      78421
# FP      12529
# FN      21273
# IDS     2203
# FRAG    2330
# TID     0.89
# LGD     1.66
# Eval time: 2618.0s

### Final results ###

# Per-class results:
#                 AMOTA   AMOTP   RECALL  MOTAR   GT      MOTA    MOTP    MT      ML      FAF     TP      FP      FN      IDS     FRAG    TID     LGD
# bicycle         0.534   1.023   0.555   0.820   1993    0.452   0.443   46      53      14.5    1097    197     887     9       12      1.06    1.59
# bus             0.573   1.225   0.667   0.774   2112    0.509   0.888   43      24      20.1    1387    313     703     22      74      0.51    1.54
# car             0.765   0.746   0.779   0.853   58317   0.649   0.580   2323    671     113.5   44360   6514    12908   1049    1101    0.43    0.98
# motorcy         0.508   1.054   0.503   0.840   1977    0.416   0.511   38      51      11.7    980     157     983     14      20      1.45    2.21
# pedestr         0.795   0.642   0.856   0.847   25423   0.696   0.529   1228    99      74.1    20894   3199    3653    876     572     0.34    0.79
# trailer         0.082   1.620   0.222   0.429   2425    0.093   0.995   12      100     30.2    525     300     1886    14      55      1.05    2.93
# truck           0.459   1.108   0.506   0.715   9650    0.357   0.734   178     232     37.1    4825    1376    4767    58      165     0.92    1.98

# Aggregated results:
# AMOTA   0.531
# AMOTP   1.060
# RECALL  0.584
# MOTAR   0.754
# GT      14556
# MOTA    0.453
# MOTP    0.669
# MT      3868
# ML      1230
# FAF     43.0
# TP      74068
# FP      12056
# FN      25787
# IDS     2042
# FRAG    1999
# TID     0.82
# LGD     1.72
# Eval time: 2801.4s