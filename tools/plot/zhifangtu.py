import torch
import math
import numpy as np
import matplotlib.pyplot as plt
model = torch.load("/workspace/S/jiangfei/BinaryNeuralNetwork_debug/work_dirs/rprelu/react_a/adreact_gprelusign_nds54_step1/latest.pth")


values = []
for k,v in model["state_dict"].items():
        if "move" in k:
                value=v.reshape(-1)
                value = value.numpy()
                values.append(value)

ncols = math.ceil(math.sqrt(len(values)))
nrows = math.ceil(len(values) / ncols)
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))

for ax, fea in zip(axs.flat, values):
        print(f'plotting img...')
        ax.hist(fea, bins=20, color='blue', alpha=0.7)
        ax.grid()
fig.savefig('./gprsign_dns54.jpg')
