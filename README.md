# Binary Neural Network

### 1、简介

本代码为二值网络模拟实验代码，基于[mmclassification](https://github.com/open-mmlab/mmclassification)框架实现。

mmclassification文档：https://mmclassification.readthedocs.io/

mmclassification安装说明：https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md

### 2、安装

​	参考mmclassification安装说明

* 1、安装torch，torchvision

* 2、git clone https://github.com/lucifer443/BinaryNeuralNetwork.git

* 3、安装依赖项

  ```
  pip install -r requirements.txt
  ```

* 4、安装mmcv

### 3、使用说明

​	参考mmclassification文档

**训练：**

```
./dist_train.sh $CONFIG_FILE $NUM_GPU
```

**推理：**

```
./dist_test.sh $CONFIG_FILE $CHECKPOINT $NUM_GPU
```

**重要文件说明**：

* configs/baseline/: baseline binary模型的config文件
* configs/binary/: 一些binary config文件
* mmcls/models/backbones/binary_backbone.py: binary 网络结构定义
* mmcls/models/backbones/binary_utils:
  * binary_functions.py: 各种激活函数，各种Sign函数
  * binary_convs.py: 各种binary conv实现
  * binary_blocks.py 各种binary block实现
* mmcls/models/classifiers/distiller.py: 使用KD Loss蒸馏的分类器
* mmcls/core/utils/ede.py： EDE功能实现

**config文件介绍：**

以configs/baseline/irnet_r18.py 为例

```python
_base_ = [
    '../_base_/datasets/imagenet_bs128.py',  # batch size 128*8
    '../_base_/schedules/imagenet_bs1024.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',  # classifier类型，支持“ImageClassifier”、
    backbone=dict(           # “DistillingImageClassifier”
        type='ResArch',      # 网络结构类型，目前只支持“ResArch”
        arch='IRNet-18',     # block 类型，支持“IRNet-18(34)”、“ReActNet-18(34)”
        num_stages=4,
        out_indices=(3, ),
        binary_type=(True, True), # 二值化类型，分别代表神经元(a)和权值(x)，默认全为True
        stem_act='hardtanh', # stem部分激活函数，支持“hardtanh”、“relu”、“prelu”和None
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='IRClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# EDE功能开启
custom_imports = dict(imports=['mmcls.core.utils.ede'], allow_failed_imports=False)
custom_hooks = [
    dict(type='EDEHook', total_epoch=100)
]
work_dir = 'work_dirs/baseline/irnet_r18'
find_unused_parameters=True
seed = 166  # 训练seed，建议设置
```

### 4、MODEL ZOO

**baseline模型**

- [x] [IR-Net](https://arxiv.org/abs/1909.10788)
- [x] [ReActNet](https://arxiv.org/abs/2003.03488)
- [x] [Bi-Real Net](https://arxiv.org/abs/1808.00278)
- [ ] [Real-To-Binary Net](https://arxiv.org/abs/2003.11535)
- [ ] [High-Capcity Expert Binary Net](https://arxiv.org/abs/2010.03558)

| 模型名称        | 复现精度 | 官方精度 | 来源论文           | 详细情况                         |
| --------------- | -------- | -------- | ------------------ | -------------------------------- |
| irnet_r18       | 58.58    | 58.1     | IR-Net             | [](configs/baseline/irnet)       |
| reactnet_r18    | 66.1     | 65.9     | ReActNet           | [](configs/baseline/reactnet)    |
| reactnet_a      | 70.0     | 69.5     | ReActNet           | [](configs/baseline/reactnet)    |
| strong_baseline | 60.45    | 60.9     | Real-To-Binary Net | [](configs/baseline/real2binary) |
| birealnet_r18   | 57.23    | 56.4     | Bi-Real Net        | [](configs/baseline/birealnet) |



### 5、功能列表

- [x] Error Decay Estimator  （EDE）
- [x] KD distilling
- [x] BOPs计算工具

### 6、使用建议和注意事项

1. baseline最好先选择irnet_r18，因为它训练时间短，可以快速迭代
2. 训练时最好设置seed，不是seed可能会导致-0.5%~+0.5%的精度差异

