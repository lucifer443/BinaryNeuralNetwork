_base_ = [
    '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResArch',
        arch='MultiBias',
        num_stages=4,
        out_indices=(3, ),
        binary_type=(True, True),
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

load_from = 'work_dirs/multi_bias_step1/epoch_75.pth'
find_unused_parameters=False
seed = 166
