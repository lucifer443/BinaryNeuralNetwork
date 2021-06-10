_base_ = [
    './mfnet_1_b11100_b32x8_step2.py'
]

# specific lr
optimizer = dict(
    paramwise_cfg = dict(
        custom_keys={
            '.layer1': dict(lr_mult=0.01),
            '.layer2': dict(lr_mult=0.01),
}))

work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_b32x8/mfnet_1_b11100_s2_frome120_b32x8_step2/mfnet_1_b11100_finetune_b32x8_step2'
load_from = 'work_dir/mfnet/mfnet_1/mfnet_1_b32x8/mfnet_1_b11000_b32x8_step2/epoch_120.pth'
