_base_ = [
    '../../../_base_/datasets/imagenet_bs32.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MFNet',
        arch='mf_1',
        binary_type=(True, True),
        stem_conv_ks=7),
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

work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_sb/mfnet_1_sb_step1'
find_unused_parameters=True
seed = 166
