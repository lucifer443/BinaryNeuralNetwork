_base_ = [
    './mfnet_1_b32x8_step2.py'
]

model = dict(
    backbone=dict(
        block_act=('prelu', 'dprelu'),
))


work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_b32x8/mfnet_1_dprelu_from_b32x8_step2'
load_from = 'work_dir/mfnet/mfnet_1/mfnet_1_b32x8/mfnet_1_b32x8_step1/epoch_256_prelu.pth'
