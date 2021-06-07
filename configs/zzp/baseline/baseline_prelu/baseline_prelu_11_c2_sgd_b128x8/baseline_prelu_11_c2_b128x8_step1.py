_base_ = [
    '../../../../_base_/datasets/imagenet_bs128.py',
    '../../../../_base_/schedules/imagenet_bs1024.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Baseline',
        arch='baseline_11',
        binary_type=(True, False),
        stem_act='prelu',
        stem_channels=64,
        base_channels=90,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=720,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


work_dir = 'work_dir/baseline/baseline_prelu/baseline_prelu_11_c2_sgd_b128x8/baseline_prelu_11_c2_sgd_b128x8_step1'
find_unused_parameters=False
seed = 166
