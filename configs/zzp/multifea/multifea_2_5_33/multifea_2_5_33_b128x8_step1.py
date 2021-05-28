_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/schedules/imagenet_bs1024.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiFea',
        arch='mf_2_5',
        binary_type=(True, False),
        stem_act='hardtanh',
        block_act=('identity', 'hardtanh'),
        thres=(-0.33, 0.33),
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='IRClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


work_dir = 'work_dir/multifea/multifea_2_5/multifea_2_5_33/multifea_2_5_33_b128x8_step1'
find_unused_parameters=False
seed = 166