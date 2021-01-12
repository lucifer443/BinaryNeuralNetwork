_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
use_fp16 = True
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BinaryResNet34',
        model_path="/workspace/AI_algorithm/binary/IR-Net/ImageNet/ResNet34/1w1a/models"),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
find_unused_parameters=True
