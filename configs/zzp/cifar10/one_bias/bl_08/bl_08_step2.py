_base_ = [
    './bl_08_step1.py',
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
))

optimizer = dict(
    weight_decay=0.0,
)

load_from = 'work_dir/cifar10/one_bias/bl_08/bl_08_step1/epoch_200.pth'
work_dir = 'work_dir/cifar10/one_bias/bl_08/bl_08_step2'
