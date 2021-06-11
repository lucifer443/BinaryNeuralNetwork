_base_ = [
    './mfnet_1_sb_step2.py'
]

model = dict(
    backbone=dict(
        binary_type_cfg=((True, False), (True, False), (True, False), (True, True), (True, True),),
))


work_dir = 'work_dir/mfnet/mfnet_1/mfnet_1_sb/mfnet_1_b00011_sb_step2'
