_base_ = [
    '../../../_base_/datasets/imagenet_bs32.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiFea',
        arch='mf_2_5',
        binary_type=(True, False),
        stem_act='prelu',
        block_act=('prelu', 'identity'),
        thres=(-0.40, 0.40),
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

# schedules for imagenet bs256
optimizer = dict(
    type='Adam',
    lr=5e-4,
    weight_decay=1e-5,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
)
runner = dict(type='EpochBasedRunner', max_epochs=256)

work_dir = 'work_dir/multifea/multifea13_2_5/multifea13_2_5_40/multifea13_2_5_40_reactnet_step1'
find_unused_parameters=False
seed = 166
