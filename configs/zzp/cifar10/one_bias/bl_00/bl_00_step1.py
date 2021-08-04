_base_ = [
    '../../../../_base_/datasets/cifar10_bs16.py',
    '../../../../_base_/schedules/cifar10_bs128.py',
    '../../../../_base_/default_runtime.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiFea_CIFAR',
        arch='mf_1_5',
        binary_type=(True, False),
        stem_act='prelu',
        block_act=('prelu', 'identity'),
        thres=(0.0,),
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

checkpoint_config = dict(interval=10)

work_dir = 'work_dir/cifar10/one_bias/bl_00/bl_00_step1'
find_unused_parameters=False
seed = 166
