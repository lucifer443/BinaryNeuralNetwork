_base_ = [
    '../../../_base_/datasets/imagenet_bs64_pil_resize.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResArch',
        arch='ReActNet-18fl',
        num_stages=4,
        out_indices=(3, ),
        binary_type=(False, False),
        stem_act='prelu',
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

optimizer = dict(
    type='Adam',
    lr=1e-3,
    weight_decay=1e-5,
    paramwise_cfg=dict(
        norm_decay_mult=0,
        custom_keys={
            # '.rebias1': dict(decay_mult=0.0),
            # '.rebias2': dict(decay_mult=0.0),
            # '.prelu1': dict(decay_mult=0.0),
            # '.prelu2': dict(decay_mult=0.0),
        }
    )
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
)
runner = dict(type='EpochBasedRunner', max_epochs=50)

work_dir = 'work_dirs/rprelu/react_a1/react18_lb_float'
find_unused_parameters=False
seed = 166