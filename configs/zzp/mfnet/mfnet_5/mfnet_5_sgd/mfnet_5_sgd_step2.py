_base_ = [
    './mfnet_5_sgd_step1.py'
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
))

optimizer = dict(
    weight_decay=0.0,
)


load_from = 'work_dir/mfnet/mfnet_5/mfnet_5_sgd/mfnet_5_sgd_step1/epoch_100.pth'
work_dir = 'work_dir/mfnet/mfnet_5/mfnet_5_sgd/mfnet_5_sgd_step2'
