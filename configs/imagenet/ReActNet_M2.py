_base_ = [
    '../_base_/datasets/imagenet_bs512.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BiReal18',
        model_path="/home/changming/codebase/binary/ReActNet/mobilenet/2_step2/"),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
find_unused_parameters=True
