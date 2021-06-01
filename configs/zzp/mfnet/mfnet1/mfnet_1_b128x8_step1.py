_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/schedules/imagenet_bs1024.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='DistillingImageClassifier',
    backbone=dict(
        type='MFNet',
        arch='mf_1',
        binary_type=(True, False)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=0.5),
        topk=(1, 5),),
    distill=dict(
        teacher_cfg='configs/_base_/models/resnet34.py',
        teacher_ckpt='work_dir/teacher_ckpts/resnet34_batch256_imagenet_20200708-32ffb4f7.pth',
        loss_weight=0.5,
        only_kdloss=False))


work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_b128x8/mfnet_1_b128x8_step1'
find_unused_parameters=True
seed = 166
