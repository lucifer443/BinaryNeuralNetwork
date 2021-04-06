_base_ = [
    '../_base_/datasets/imagenet_bs128.py',
    '../_base_/schedules/imagenet_bs1024.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DistillingImageClassifier',
    backbone=dict(
        type='ResArch',
        arch='IRNet-18-bias',
        num_stages=4,
        out_indices=(3, ),
        stem_act='hardtanh',
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='IRClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    distill=dict(
        teacher_cfg='configs/_base_/models/resnet34.py',
        teacher_ckpt='/lustre/S/jiangfei/BinaryNeuralNetwork/work_dirs/resnet34_batch256_imagenet_20200708-32ffb4f7.pth',
        loss_weight=1.,
        only_kdloss=True))

custom_imports = dict(imports=['mmcls.core.utils.ede'], allow_failed_imports=False)
custom_hooks = [
    dict(type='EDEHook', total_epoch=100)
]

work_dir = 'work_dirs/irnet_r18_distill'
find_unused_parameters=True
seed = 166
