_base_ = [
    '../../../_base_/datasets/imagenet_bs64_pil_resize.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileArch',
        arch='ReActNet-A',
        Expand_num = 1.2,
        rpgroup = 1,
        gp = 16,
        binary_type=(True, False),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

optimizer = dict(
    type='Adam',
    lr=2e-3,
    weight_decay=1e-5,
    paramwise_cfg=dict(
        norm_decay_mult=0,
        custom_keys={
            #'.stem_act': dict(decay_mult=0.0),
            #'.move1': dict(decay_mult=0.0),
            '.adbias1': dict(lr_mult=0.0,lr_add=1e-5,decay_mult=0.0),
            '.adbias2': dict(lr_mult=0.0,lr_add=1e-5,decay_mult=0.0),
            '.prelu1': dict(decay_mult=0.0),
            '.prelu2': dict(decay_mult=0.0),
        }
    )
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=25025,
    warmup_ratio=0.1,
    step=[40, 60, 70],
)
runner = dict(type='EpochBasedRunner', max_epochs=75)
work_dir = 'work_dirs/rprelu/react_a1/adreact_3t1_0fe-5_step1'
find_unused_parameters=False
seed = 166