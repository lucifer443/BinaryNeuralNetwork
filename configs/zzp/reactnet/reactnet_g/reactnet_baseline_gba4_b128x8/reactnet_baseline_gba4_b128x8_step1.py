_base_ = [
    '../../../../_base_/datasets/imagenet_bs64.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ReActNet',
        arch='reactnet_baseline_gba4',
        binary_type=(True, False)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings for bs128
data = dict(
    workers_per_gpu=2,
)

# schedules for imagenet bs256
optimizer = dict(
    type='Adam',
    lr=5e-4,
    weight_decay=1e-5,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
)
runner = dict(type='EpochBasedRunner', max_epochs=256)

work_dir = 'work_dir/reactnet/reactnet_g/reactnet_baseline_gba4/reactnet_baseline_gba4_b128x8_step1'
find_unused_parameters=True
seed = 166
