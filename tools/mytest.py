import torch
from mmcls.models import build_classifier
from mmcv import Config



cfg = Config.fromfile("/lustre/S/jiangfei/BinaryNeuralNetwork/configs/baseline/reactnet_A_step1.py")
model = build_classifier(cfg.model)
print(model)