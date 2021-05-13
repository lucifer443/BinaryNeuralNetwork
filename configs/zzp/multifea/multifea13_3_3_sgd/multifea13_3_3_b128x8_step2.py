_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/schedules/imagenet_bs1024.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Baseline',
        arch='mf13_3_3',
        binary_type=(True, True),
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


load_from = 'work_dir/multifea/multifea13_3_3_sgd/multifea13_3_3_b128x8/multifea13_3_3_b128x8_step1/epoch_75.pth'
work_dir = 'work_dir/multifea/multifea13_3_3_sgd/multifea13_3_3_b128x8/multifea13_3_3_b128x8_step2'
find_unused_parameters=False
seed = 166
