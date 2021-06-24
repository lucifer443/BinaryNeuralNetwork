_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/schedules/imagenet_bs1024.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiFea',
        arch='mf_2_5',
        binary_type=(False, False),
        stem_act='prelu',
        block_act=('prelu', 'identity'),
        thres=(-0.55, 0.55),
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


work_dir = 'work_dir/multifea/multifea13_2_5/multifea13_2_5_55/multifea13_2_5_55_b128x8_float'
find_unused_parameters=False
seed = 166