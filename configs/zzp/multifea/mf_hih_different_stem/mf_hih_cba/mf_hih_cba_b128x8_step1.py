_base_ = [
    '../../../../_base_/datasets/imagenet_bs128.py',
    '../../../../_base_/schedules/imagenet_bs1024.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiFea',
        arch='mf_1_1',
        binary_type=(True, False),
        stem_order='cba',
        strides=(2, 2, 2, 2),
        stem_act='hardtanh',
        block_act=('identity', 'hardtanh'),
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


work_dir = 'work_dir/multifea/mf_hih_different_stem/mf_hih_cba/mf_hih_cba_b128x8_step1'
find_unused_parameters=False
seed = 166
