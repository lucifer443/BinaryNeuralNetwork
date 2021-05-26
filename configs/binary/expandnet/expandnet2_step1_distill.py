_base_ = [
    '../../_base_/datasets/imagenet_bs64.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='DistillingImageClassifier',
    backbone=dict(
        type='MobileArch',
        arch='ExpandNet-2',
        stem_channels=24,
        binary_type=(True, False),
        stem_act='hardtanh'),
    neck=dict(type='GlobalAveragePoolingBN', in_channels=768),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=0.5),
        topk=(1, 5),),
    distill=dict(
        teacher_cfg='configs/_base_/models/resnet34.py',
        teacher_ckpt='work_dirs/resnet34_batch256_imagenet_20200708-32ffb4f7.pth',
        loss_weight=0.5,
        only_kdloss=False))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25,
    step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

find_unused_parameters=False
seed = 166
