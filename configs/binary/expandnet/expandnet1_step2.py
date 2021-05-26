_base_ = [
    '../../_base_/datasets/imagenet_bs64.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileArch',
        arch='ExpandNet-1',
        binary_type=(True, True),
        stem_act='hardtanh'),
    neck=dict(type='GlobalAveragePoolingBN', in_channels=1024),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9, weight_decay=0., nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25,
    step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

load_from = 'work_dirs/expandnet1_step1/epoch_100.pth'
find_unused_parameters=False
seed = 166