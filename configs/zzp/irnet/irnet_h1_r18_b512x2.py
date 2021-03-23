_base_ = [
    '../../_base_/datasets/imagenet_bs128.py',
    '../../_base_/schedules/imagenet_bs1024.py', '../../_base_/default_runtime.py'
]

# datasets settings for bs512
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=1)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='IRNet',
        arch='irnet_h1_r18',
        stem_channels=64,
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
        topk=(1, 5),
    ))

custom_imports = dict(imports=['mmcls.core.utils.ede'], allow_failed_imports=False)
custom_hooks = [dict(type='EDEHook', total_epoch=100)]

work_dir = 'work_dir/irnet_h1_r18_b512x2'
find_unused_parameters=True
seed = 166
