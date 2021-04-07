_base_ = [
    '../../../../_base_/datasets/imagenet_bs32.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ReActNet',
        arch='reactnet_baseline',
        stem_channels=32,
        binary_type=(True, False)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

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

work_dir = 'work_dir/reactnet/reactnet_baseline/reactnet_baseline_b32x8/reactnet_baseline_b32x8_step1'
find_unused_parameters=False
seed = 166
