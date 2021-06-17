_base_ = [
    './mfnet_1_sgd_step1.py'
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
        block_act=('prelu', 'scale_one'),
))

optimizer = dict(
    weight_decay=0.0,
)


load_from = 'work_dir/mfnet/mfnet_1/mfnet_1_sgd/mfnet_1_sgd_step1/epoch_100_rprelu.pth'
work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_sgd/mfnet_1_scale_one_fromsgd_step2'
