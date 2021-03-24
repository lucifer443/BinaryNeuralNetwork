_base_ = [
    '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='reactnet_A',
        binary_type=(True, False)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),))

optimizer = dict(
    type='Adam',
    lr=5e-4,
    weight_decay=0.00001,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
)
runner = dict(type='EpochBasedRunner', max_epochs=256)

work_dir = '/lustre/S/jiangfei/BinaryNeuralNetwork/work_dirs/react_A_32/a100_s1'
find_unused_parameters=True
seed = 166
