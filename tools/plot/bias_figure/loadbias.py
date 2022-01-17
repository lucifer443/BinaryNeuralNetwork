import sys
import torch
sys.path.append('/workspace/S/jiangfei/BinaryNeuralNetwork_debug')

models = torch.load('/workspace/S/jiangfei/BinaryNeuralNetwork_debug/work_dirs/rprelu/react_a1/adreact_af75_fl_lb/epoch_45.pth')
w=[]
model = models["state_dict"]

for key,v in list(model.items()):
    if 'rebias' not in key:
        #breakpoint()
        del model[key]
for key,v  in model.items():
    #breakpoint
    if 'rebias' not in key:
        #breakpoint()
        print(key)
torch.save(models,'/workspace/S/jiangfei/BinaryNeuralNetwork_debug/work_dirs/rprelu/react_a1/adreact_af75_fl_lb/lb.pth')

aa = torch
models = torch.load('/workspace/S/jiangfei/BinaryNeuralNetwork_debug/work_dirs/rprelu/react_a1/adreact_af75_fl_lb/lb.pth')
model = models["state_dict"]
for key,v  in model.items():
    #breakpoint
    print(v)