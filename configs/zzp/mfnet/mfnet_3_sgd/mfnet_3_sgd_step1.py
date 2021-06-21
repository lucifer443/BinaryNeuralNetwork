_base_ = [
    '../../../_base_/datasets/imagenet_bs128.py',
    '../../../_base_/schedules/imagenet_bs1024.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MFNet',
        arch='mf_3',
        binary_type=(True, False),
        block_act=('prelu', 'prelu_pi=1'),
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
            '.stem_act': dict(decay_mult=0.0),
            '.nonlinear11': dict(decay_mult=0.0),
            '.nonlinear12': dict(decay_mult=0.0),
            '.nonlinear21': dict(decay_mult=0.0),
            '.nonlinear22': dict(decay_mult=0.0),
}))

work_dir = 'work_dir/mfnet/mfnet_3/mfnet_3_sgd/mfnet_3_sgd_step1'
find_unused_parameters=True
seed = 166
