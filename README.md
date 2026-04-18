# Tiny ImageNet + ResNet-18 + FGSM/PGD 对抗攻击实验

本项目用于完成 Tiny ImageNet 数据集上的图像分类模型训练，并为后续
FGSM、PGD 对抗攻击实验提供模型和评估脚本。

实验配置：

- 数据集：Tiny ImageNet，200 类。
- 模型：ResNet-18。
- 输入尺寸：64 x 64。
- 训练输出：验证集 top-1、top-5、loss、checkpoint、质量报告。
- 攻击方法：FGSM 和 PGD-Linf。

## 项目结构

```text
.
├── train.py                 # ResNet-18 训练脚本
├── evaluate_attacks.py      # clean / FGSM / PGD 评估脚本
├── src/
│   ├── tiny_imagenet.py     # Tiny ImageNet 数据加载
│   ├── resnet.py            # 64x64 输入适配版 ResNet-18
│   ├── attacks.py           # FGSM / PGD 实现
│   └── metrics.py           # accuracy 和日志工具
├── pyproject.toml           # uv 项目依赖配置
├── uv.lock                  # uv 锁文件
└── requirements.txt         # pip/汇报用依赖清单
```

## 数据集

Tiny ImageNet 数据集来自 Stanford CS231n 发布地址：

```bash
mkdir -p data
curl -L http://cs231n.stanford.edu/tiny-imagenet-200.zip \
  -o data/tiny-imagenet-200.zip
unzip -q data/tiny-imagenet-200.zip -d data
```

当前工作区中已下载并解压到：

```text
data/tiny-imagenet-200
```

数据集检查结果：

- 类别数：200
- 训练集图片：100000
- 验证集图片：10000
- 验证集标注：10000 行
- 解压后大小：475M
- 压缩包大小：241M
- 压缩包 SHA-256：

```text
6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821
```

目录结构应为：

```text
data/tiny-imagenet-200/
  train/
  val/
    images/
    val_annotations.txt
  test/
  wnids.txt
```

## 环境安装

推荐使用 uv：

```bash
uv sync --locked
```

pip 安装命令：

```bash
python3 -m pip install -r requirements.txt
```

## 快速检查

正式训练前，建议先跑 1 个 epoch 确认数据、模型和环境都正常：

```bash
uv run --locked python train.py \
  --data-root data/tiny-imagenet-200 \
  --output-dir runs/debug \
  --epochs 1 \
  --batch-size 16
```

## 模型训练

正式训练命令：

```bash
uv run --locked python train.py \
  --data-root data/tiny-imagenet-200 \
  --output-dir runs/resnet18_tinyimagenet \
  --epochs 90 \
  --batch-size 128 \
  --lr 0.1 \
  --amp
```

CPU 或小显存训练命令：

```bash
uv run --locked python train.py \
  --data-root data/tiny-imagenet-200 \
  --output-dir runs/resnet18_tinyimagenet \
  --epochs 90 \
  --batch-size 64 \
  --lr 0.1
```

训练输出：

- `runs/resnet18_tinyimagenet/best.pt`：验证集 top-1 最优 checkpoint。
- `runs/resnet18_tinyimagenet/last.pt`：最后一个 epoch 的 checkpoint。
- `runs/resnet18_tinyimagenet/metrics.csv`：每个 epoch 的 loss、top-1、top-5。
- `runs/resnet18_tinyimagenet/quality_report.json`：结构化模型质量报告。
- `runs/resnet18_tinyimagenet/quality_report.md`：可直接放入汇报材料的指标表。

## Kaggle 免费 GPU 训练

在 Kaggle 新建 Notebook 后，右侧 `Settings` 中设置：

- `Accelerator`：`GPU T4 x2`
- `Internet`：`On`

添加 Tiny ImageNet 数据集：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet
```

本项目的 GitHub 地址：

```text
https://github.com/liangliguo/imgnet
```

训练使用的数据根目录：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200
```

在 Kaggle Notebook 中依次执行下面的单元。

检查 GPU：

```python
!nvidia-smi

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

克隆代码：

```python
!git clone https://github.com/liangliguo/imgnet.git
%cd imgnet
```

安装依赖：

```python
!pip install -q -r requirements.txt
```

快速检查训练流程：

```python
!python train.py \
  --data-root /kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200 \
  --output-dir /kaggle/working/runs/debug \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --data-parallel \
  --amp
```

训练启动后会输出：

```text
device=cuda
cuda_available=True
cuda_device_count=2
data_root=/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200
output_dir=/kaggle/working/runs/debug
building_dataloaders=start
building_dataloaders=done
train_images=100000
val_images=10000
num_classes=200
data_parallel=true
epoch 1/1 lr=0.100000
start train epoch 1
```

正式训练：

```python
!python train.py \
  --data-root /kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200 \
  --output-dir /kaggle/working/runs/resnet18_tinyimagenet \
  --epochs 90 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 0.1 \
  --data-parallel \
  --amp
```

FGSM 和 PGD 评估：

```python
!python evaluate_attacks.py \
  --data-root /kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200 \
  --checkpoint /kaggle/working/runs/resnet18_tinyimagenet/best.pt \
  --output /kaggle/working/runs/resnet18_tinyimagenet/attack_report.json \
  --eps 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10
```

打包训练输出：

```python
!zip -r /kaggle/working/resnet18_tinyimagenet_outputs.zip \
  /kaggle/working/runs/resnet18_tinyimagenet
```

Kaggle 路径约定：

- `/kaggle/input`：只读，用于读取数据集。
- `/kaggle/working`：可写，用于保存 checkpoint、日志和报告。

## 可汇报的模型质量指标

训练完成后，重点汇报以下指标：

- 最优验证集 top-1 accuracy。
- 最优验证集 top-5 accuracy。
- 最优验证集 cross-entropy loss。
- 训练集和验证集 accuracy/loss 曲线。
- 模型参数量。
- 训练配置：epoch、batch size、learning rate、optimizer、weight decay。

这些信息会自动写入：

```text
runs/resnet18_tinyimagenet/quality_report.md
runs/resnet18_tinyimagenet/quality_report.json
runs/resnet18_tinyimagenet/metrics.csv
```

## 对抗攻击评估

模型训练完成后，使用最优 checkpoint 做 clean、FGSM、PGD 评估：

```bash
uv run --locked python evaluate_attacks.py \
  --data-root data/tiny-imagenet-200 \
  --checkpoint runs/resnet18_tinyimagenet/best.pt \
  --eps 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10
```

其中：

- `--eps 8/255`：L-infinity 扰动预算，按原始像素尺度计算。
- `--pgd-alpha 2/255`：PGD 每一步的步长。
- `--pgd-steps 10`：PGD 迭代步数。

输出文件：

```text
runs/resnet18_tinyimagenet/attack_report.json
```

该文件包含 clean、FGSM、PGD 下的 top-1、top-5 和 loss，可用于比较模型在正常样本和对抗样本上的性能下降。
