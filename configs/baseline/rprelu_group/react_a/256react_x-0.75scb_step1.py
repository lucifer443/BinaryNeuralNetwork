_base_ = [
    '../../../_base_/datasets/imagenet_bs32.py', '../../../_base_/default_runtime.py'
]

model = dict(
    type='DistillingImageClassifier',
    backbone=dict(
        type='MobileArch',
        arch='ReActNet-E',
        Expand_num = 0.75,
        rpgroup = 1,
        gp = 1,
        binary_type=(True, False),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=0.5),
        topk=(1, 5),),
     distill=dict(
        teacher_cfg='/workspace/S/jiangfei/BinaryNeuralNetwork_debug/configs/_base_/models/resnet34.py',
        teacher_ckpt='/workspace/S/jiangfei/BinaryNeuralNetwork_debug/work_dirs/resnet34_batch256_imagenet_20200708-32ffb4f7.pth',
        loss_weight=0.5,
        only_kdloss=False)
        )

# schedules for imagenet bs256
optimizer = dict(
    type='Adam',
    lr=5e-4,
    weight_decay=1e-5,
    paramwise_cfg=dict(
        norm_decay_mult=0,
        custom_keys={
            '.prelu1': dict(decay_mult=0.0),
            '.prelu2': dict(decay_mult=0.0),},
    )
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
)
runner = dict(type='EpochBasedRunner', max_epochs=256)



work_dir = 'work_dirs/rprelu/react_a/256react_lb1_scb_step1'
load_from = 'work_dirs/rprelu/react_a1/adreact_lf_gw_lb/epoch_1.pth'
#resume_from = 'work_dirs/rprelu/react_a/adreact_rprelu_step1/latest.pth'
find_unused_parameters=False
seed = 166