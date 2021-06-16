_base_ = [
    '../../_base_/datasets/imagenet_bs32.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RPreluArch',
        arch='RPrereact',
        num_stages=4,
        out_indices=(3, ),
        Expand_num = 1,
        rpgroup = 2,
        gp = 2,
        binary_type=(True, False),
        stem_act='prelu',
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

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


work_dir = 'work_dirs/rprelu/gpreact18_step1_s3_2'
find_unused_parameters=False
seed = 166