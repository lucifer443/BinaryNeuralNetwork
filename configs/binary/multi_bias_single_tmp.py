_base_ = [
    '../_base_/datasets/imagenet_bs128.py',
    '../_base_/schedules/imagenet_bs1024.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResArch',
        arch='MultiBias',
        num_stages=4,
        out_indices=(3, ),
        stem_act='hardtanh',
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='IRClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
# optimizer
optimizer = dict(
    type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25,
    step=[30, 60, 90, 95])
runner = dict(type='EpochBasedRunner', max_epochs=100)

find_unused_parameters=False
seed = 166