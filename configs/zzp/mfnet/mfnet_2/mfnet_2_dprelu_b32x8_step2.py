_base_ = [
    './mfnet_2_dprelu_b32x8_step1.py'
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
))

optimizer = dict(
    type='Adam',
    lr=5e-4,
    weight_decay=0.0,
    paramwise_cfg=dict(norm_decay_mult=0)
)


load_from = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_step1/epoch_256.pth'
work_dir = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_step2'
