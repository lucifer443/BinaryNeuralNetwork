_base_ = [
    '../../../../_base_/datasets/imagenet_bs128.py',
    '../../../../_base_/schedules/imagenet_bs1024.py', '../../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MFNet',
        arch='mf_5',
        binary_type=(False, False),
        fea_num=1,
        fexpand_mode='5',
        thres=(0.6,),
        block_act=('prelu', 'dprelu'),
        stem_conv_ks=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
))

optimizer = dict(
    paramwise_cfg = dict(
        custom_keys={
            '.nonlinear12': dict(decay_mult=0.0),
            '.nonlinear22': dict(decay_mult=0.0),
}))

lr_config = dict(
    warmup_ratio=0.1,
)

# log_config = dict(
#     interval=1,
# )


work_dir = 'work_dir/mfnet/mfnet_5/mfnet_5_sgd/mfnet_5_bias06_float'
find_unused_parameters=True
seed = 166
