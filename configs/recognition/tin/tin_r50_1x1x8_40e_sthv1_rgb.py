_base_ = ['../../_base_/models/tin_r50.py', '../../_base_/default_runtime.py']

# model settings
model = dict(cls_head=dict(num_classes=174, dropout_ratio=0.8))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/sthv1/rawframes'
data_root_val = 'data/sthv1/rawframes'
ann_file_train = 'data/sthv1/sthv1_train_list_rawframes.txt'
ann_file_val = 'data/sthv1/sthv1_val_list_rawframes.txt'
ann_file_test = 'data/sthv1/sthv1_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

val_cfg = dict(interval=1)
test_cfg = dict()

# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.02,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0005)
default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=20, norm_type=2)))

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        T_max=39,
        eta_min=0,
        by_epoch=True,
        begin=1,
        end=40)
]

train_cfg = dict(by_epoch=True, max_epochs=140)
