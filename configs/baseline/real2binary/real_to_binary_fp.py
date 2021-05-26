_base_ = [
    '../../_base_/datasets/imagenet_bs32_colorjitter.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='ATKDImageClassifier',
    backbone=dict(
        type='ResArch',
        arch='Real2BiFP',
        num_stages=4,
        out_indices=(3, ),
        binary_type=(False, False),
        stem_act='prelu',
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),),
    distill=dict(
        teacher_cfg='configs/_base_/models/resnet18.py',
        teacher_ckpt='pretrained/resnet18_batch256_imagenet_20200708-34ab8f90.pth',
        kd_weight=3,
        at_weight=30.,
        ce_weight=1.))

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

find_unused_parameters=False
seed = 166