_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/schedules/imagenet_bs1024.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MFNet',
        arch='mf_1',
        binary_type=(False, False),
        stem_conv_ks=7),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
))

log_config = dict(
    interval=1,
)

optimizer = dict(lr=0.1)
lr_config = dict(warmup_ratio=0.1)


work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_sgd/mfnet_1_sgd_float'
find_unused_parameters=True
seed = 166
