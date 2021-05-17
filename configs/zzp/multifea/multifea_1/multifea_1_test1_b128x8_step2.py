_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Baseline',
        arch='mf_1',
        binary_type=(True, True),
        stem_act='hardtanh',
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

# schedules for bs1024
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.4,
    momentum=0.9,
    weight_decay=0,
    nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

load_from = 'work_dir/multifea/multifea_1/multifea_1_b128x8_two/multifea_1_b128x8_step1/epoch_100.pth'
work_dir = 'work_dir/multifea/multifea_1/multifea_1_b128x8_two/multifea_1_test1_b128x8_step2'
find_unused_parameters=False
seed = 166
