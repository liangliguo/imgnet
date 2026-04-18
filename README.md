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

如果不用 uv，也可以使用 `requirements.txt`：

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

如果没有 CUDA 或显存不足，可以去掉 `--amp`，并适当减小 `--batch-size`。

训练输出：

- `runs/resnet18_tinyimagenet/best.pt`：验证集 top-1 最优 checkpoint。
- `runs/resnet18_tinyimagenet/last.pt`：最后一个 epoch 的 checkpoint。
- `runs/resnet18_tinyimagenet/metrics.csv`：每个 epoch 的 loss、top-1、top-5。
- `runs/resnet18_tinyimagenet/quality_report.json`：结构化模型质量报告。
- `runs/resnet18_tinyimagenet/quality_report.md`：可直接放入汇报材料的指标表。

## Kaggle 免费 GPU 训练

Kaggle Notebook 可以免费使用 GPU，但每周额度、单次运行时长和可用 GPU
型号会随 Kaggle 当前策略变化。实际训练前，以 Notebook 右侧
`Settings -> Accelerator` 中能选择到的 GPU 为准。

本项目的 GitHub 地址：

```text
https://github.com/liangliguo/imgnet
```

Kaggle 数据集路径：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet
```

推荐流程：

1. 在 Kaggle 新建 Notebook。
2. 点击右侧 `Settings`：
   - `Accelerator` 选择 `GPU`。
   - 打开 `Internet`，用于从 GitHub 克隆代码。
3. 添加 Tiny ImageNet 数据集，确保 Notebook 中能访问：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet
```

在 Kaggle Notebook 中先检查 GPU：

```python
!nvidia-smi

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

克隆本项目代码：

```python
!git clone https://github.com/liangliguo/imgnet.git
%cd imgnet
```

Kaggle 通常已经预装 PyTorch。如果依赖缺失，再执行：

```python
!pip install -q -r requirements.txt
```

查看 Kaggle 数据集挂载路径：

```python
!find /kaggle/input/datasets/akash2sharma/tiny-imagenet -maxdepth 3 -type d | head -50
```

训练脚本要求 `--data-root` 指向包含 `train/`、`val/`、`wnids.txt` 的目录。
如果上面的 `find` 输出显示这些目录就在下面这个路径中：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet
```

则训练命令为：

```python
!python train.py \
  --data-root /kaggle/input/datasets/akash2sharma/tiny-imagenet \
  --output-dir /kaggle/working/runs/resnet18_tinyimagenet \
  --epochs 90 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 0.1 \
  --amp
```

如果 `find` 输出显示实际结构是：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/train
/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/val
```

则把训练命令里的 `--data-root` 改为：

```text
/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200
```

如果显存不足，把 `--batch-size 128` 改成 `64` 或 `32`。

Kaggle 中路径约定：

- `/kaggle/input`：只读，用于读取数据集。
- `/kaggle/working`：可写，用于保存 checkpoint、日志和报告。

训练结束后，建议打包输出：

```python
!zip -r /kaggle/working/resnet18_tinyimagenet_outputs.zip \
  /kaggle/working/runs/resnet18_tinyimagenet
```

然后在 Notebook 右侧输出区域下载：

```text
resnet18_tinyimagenet_outputs.zip
```

如果一次免费 GPU 时间不够，可以把上一次的 `last.pt` 作为 Kaggle Dataset
加入下一次 Notebook，然后用 `--resume` 继续训练：

```python
!python train.py \
  --data-root /kaggle/input/datasets/akash2sharma/tiny-imagenet \
  --output-dir /kaggle/working/runs/resnet18_tinyimagenet \
  --epochs 90 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 0.1 \
  --amp \
  --resume /kaggle/input/<previous-run>/last.pt
```

在 Kaggle 上训练完成后，可以直接评估 FGSM 和 PGD：

```python
!python evaluate_attacks.py \
  --data-root /kaggle/input/datasets/akash2sharma/tiny-imagenet \
  --checkpoint /kaggle/working/runs/resnet18_tinyimagenet/best.pt \
  --output /kaggle/working/runs/resnet18_tinyimagenet/attack_report.json \
  --eps 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10
```

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
