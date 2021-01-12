_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BiReal18',
        model_path="/workspace/AI_algorithm/binary/ReActNet/resnet/2_step2/"),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
find_unused_parameters=True
