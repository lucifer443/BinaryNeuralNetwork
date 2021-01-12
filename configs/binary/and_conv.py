_base_ = [
    '../_base_/datasets/imagenet_bs128.py',
    '../_base_/schedules/imagenet_bs1024.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResArch',
        arch='IRNet-18',
        num_stages=4,
        out_indices=(1,2,3),
        style='pytorch'),
    neck=dict(
        type='MultiLevelFuse',
        in_channels=[128, 256, 512],
        out_channels=256,
        conv_type='AND'),
    head=dict(
        type='IRClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
find_unused_parameters=True
seed = 166
