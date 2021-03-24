_base_ = [
    '../_base_/datasets/imagenet_bs128.py', '../_base_/default_runtime.py'
]

model = dict(
    type='DistillingImageClassifier',
    backbone=dict(
        type='reactnet_A',
        binary_type=(True, True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),),
    distill=dict(
        teacher_cfg='configs/_base_/models/resnet34.py',
        teacher_ckpt='/lustre/S/jiangfei/BinaryNeuralNetwork/work_dirs/resnet34_batch256_imagenet_20200708-32ffb4f7.pth',
        loss_weight=1.,
        only_kdloss=True))

optimizer = dict(
    type='Adam',
    lr=4*5e-4,
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

load_from = '/lustre/S/jiangfei/BinaryNeuralNetwork/work_dirs/react_A_128/a100_s1/epoch_256.pth'
work_dir = '/lustre/S/jiangfei/BinaryNeuralNetwork/work_dirs/react_A_128/a100_s2'
find_unused_parameters=True
seed = 166
