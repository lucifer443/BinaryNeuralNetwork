_base_ = [
    '../../../../_base_/datasets/imagenet_bs128.py',
    '../../../../_base_/schedules/imagenet_bs1024.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='IRNet',
        arch='irnet_gbb_r18',
        stage_setting=(1, 2, 6, 2),
        group_cfg=((4,) * 1, (8,) * 2, (8,) * 6, (16,) * 2),
        branch_cfg=((4,) * 1, (4,) * 2, (4,) * 6, (4,) * 2),
        stem_channels=64,
        base_channels=64,
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

work_dir = 'work_dir/irnet/irnet_g/irnet_gbb/irnet_gbb_2222_48816_4444_r18_b128x8'
find_unused_parameters=True
seed = 166
