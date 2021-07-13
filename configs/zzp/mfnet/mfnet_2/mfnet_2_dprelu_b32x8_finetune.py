_base_ = [
    './mfnet_2_dprelu_b32x8_step2.py'
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
))

optimizer = dict(
    type='Adam',
    lr=1e-5,
    weight_decay=0.0,
    paramwise_cfg=dict(norm_decay_mult=0)
)
# learning policy
lr_config = dict(
    policy='fixed',
)


load_from = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_step2/epoch_250.pth'
work_dir = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_finetune'
