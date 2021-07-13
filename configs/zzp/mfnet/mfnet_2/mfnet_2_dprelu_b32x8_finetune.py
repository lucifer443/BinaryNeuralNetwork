_base_ = [
    './mfnet_2_dprelu_b32x8_step2.py'
]


optimizer = dict(
    lr=1e-5,
)
# learning policy
lr_config = dict(
    _delete_ = True,
    policy='fixed',
)


load_from = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_step2/epoch_250.pth'
work_dir = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_finetune'
