_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BinaryResNet18',
        steps_per_epoch=5004,
        total_epochs=100,
        model_path="/workspace/AI_algorithm/binary/IR-Net/ImageNet/ResNet18/1w1a_EDE/models/"),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
find_unused_parameters=True
