_base_ = [
    '../../../../_base_/datasets/imagenet_bs32.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Baseline',
        arch='baseline_11',
        binary_type=(True, False),
        stem_act='prelu',
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',),
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
    lr=1e-3,
    weight_decay=1e-5,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=25025,
    warmup_ratio=0.1,
    step=[40, 60, 70],
)
runner = dict(type='EpochBasedRunner', max_epochs=75)

work_dir = 'work_dir/baseline/baseline_prelu/baseline_prelu_11_b32x8/baseline_prelu_11_b32x8_step1'
find_unused_parameters=False
seed = 166
