_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BinaryResNet18',
        model_path="/workspace/AI_algorithm/binary/IR-Net/ImageNet/ResNet18/1w1a_EDE/models/"),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
find_unused_parameters=True
custom_imports = dict(imports=['mmdet.core.utils.ede'], allow_failed_imports=False)
custom_hooks = [
    dict(type='EDEHook', total_epoch=112)
]
