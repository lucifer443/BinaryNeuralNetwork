_base_ = [
    '../../../_base_/datasets/imagenet_bs32.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Baseline',
        arch='baseline_22',
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        binary_type=(True, True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# schedules for imagenet bs256
optimizer = dict(
    type='Adam',
    lr=2e-4,
    weight_decay=0,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[40, 60, 70],
)
runner = dict(type='EpochBasedRunner', max_epochs=75)

load_from = 'work_dir/baseline/baseline_22_b32x8/baseline_22_b32x8_step1/epoch_75.pth'
work_dir = 'work_dir/baseline/baseline_22_b32x8/baseline_22_b32x8_step2'
find_unused_parameters=False
seed = 166
