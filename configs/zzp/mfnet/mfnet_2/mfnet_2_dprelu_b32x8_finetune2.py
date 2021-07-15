_base_ = [
    './mfnet_2_dprelu_b32x8_step2.py'
]

optimizer = dict(
    lr=5e-4,
)
# learning policy
lr_config = dict(
    _delete_ = True,
    policy='step',
    gamma=0.02,
    step=[200,],
)

runner = dict(
    max_epochs=512,
)


# load_from = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_step2/epoch_250.pth'
work_dir = 'work_dir/mfnet/mfnet_2/mfnet_2_dprelu_b32x8/mfnet_2_dprelu_b32x8_finetune2'
