_base_ = './rotated_imted_vb1_oriented_rcnn_vit_base_1x_dota_le90_16h.py'

model = dict(
    backbone=dict(
        use_checkpoint=False, # True, # False for A100
    ),
    roi_head=dict(
        bbox_head=dict(
            type='RotatedMAEBBoxHeadSTDC',
            dc_mode_str_list = ['', '', '', 'XY', '', 'A', '', 'WH'],
            num_convs_list   = [0, 0, 3, 3, 2, 2, 1, 1],
            am_mode_str_list = ['', '', 'V', 'V', 'V', 'V', 'V', 'V'],
            rois_mode        = 'rbbox',
            use_checkpoint=False, # True, # False for A100
        ),
    ),
)

# dota_ms_rr
data_root_rich = '/HDD/datasets/projects/rich/24.06.19/split_dataset_box_dota/'
data_root_ms = '/HDD/datasets/public/split_ms_dota/'
data_root_ss = '/HDD/datasets/public/split_ss_dota/'
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(768, 768)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
classes = ('BOX',)
dataset_type = 'CustomDOTADataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2, # 4, # 8 for A100
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_rich + 'train/labelTxt/',
        img_prefix=data_root_rich + 'train/images/',
        pipeline=train_pipeline, version=angle_version),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_rich + 'val/labelTxt/',
        img_prefix=data_root_rich + 'val/images/',
        version=angle_version),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_rich + 'val/images/',
        img_prefix=data_root_rich + 'val/images/',
        version=angle_version))

# classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank', 'soccer-ball-field',
#                'roundabout', 'harbor', 'swimming-pool', 'helicopter')
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=2, # 4, # 8 for A100
#     train=dict(
#         classes=classes,
#         ann_file=data_root_ss + 'train/annfiles/',
#         img_prefix=data_root_ss + 'train/images/',
#         pipeline=train_pipeline, version=angle_version),
#     val=dict(
#         classes=classes,
#         ann_file=data_root_ss + 'val/annfiles/',
#         img_prefix=data_root_ss + 'val/images/',
#         version=angle_version),
#     test=dict(
#         classes=classes,
#         ann_file=data_root_ss + 'test/images/',
#         img_prefix=data_root_ss + 'test/images/',
#         version=angle_version))

# optimizer
# optimizer = dict(lr=5e-5)  # 4 GPUs for A100
