_base_ = [
    '../../_base_/datasets/imagenet_bs32.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='DistillingImageClassifier',
    backbone=dict(
        type='ReActNet',
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
        teacher_ckpt='work_dir/resnet34_batch256_imagenet_20200708-32ffb4f7.pth',
        loss_weight=1.,
        only_kdloss=True))

# schedules for imagenet bs256
optimizer = dict(
    type='Adam',
    lr=5e-4,
    weight_decay=0.,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
)
runner = dict(type='EpochBasedRunner', max_epochs=256)

work_dir = 'work_dir/reactnet_a_b32x8_step2_a100'
load_from = 'work_dir/reactnet_a_b32x8_step1_a100/epoch_256.pth'
find_unused_parameters=True
seed = 166
