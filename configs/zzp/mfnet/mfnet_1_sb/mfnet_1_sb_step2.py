_base_ = [
    './mfnet_1_sb_step1.py'
]

model = dict(
    backbone=dict(
        binary_type=(True, True),
))

# schedules for imagenet bs256
optimizer = dict(
    lr=2e-4,
    weight_decay=0.0,
)
optimizer_config = dict(grad_clip=None)

load_from = 'work_dir/mfnet/mfnet_1/mfnet_1_sb/mfnet_1_sb_step1/epoch_75.pth'
work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_sb/mfnet_1_sb_step2'
