_base_ = [
    './mfnet_1_ds_wd0_step1.py'
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
))

optimizer = dict(
    weight_decay=0.0,
)


load_from = 'work_dir/mfnet/mfnet_1/mfnet_1_sgd/mfnet_1_ds_wd0_step1/epoch_100.pth'
work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_sgd/mfnet_1_ds_wd0_step2'
