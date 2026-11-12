# 基于 SE-ResNet18 的 CIFAR-10 图像分类与场景应用

本项目为《神经网络与深度学习》期末大作业，实现了经典 ResNet18 与加入 SE（Squeeze-and-Excitation）通道注意力的 SE-ResNet18，并在 CIFAR-10 上完成单变量对比实验、测试集评价和真实照片场景应用。

## 1. 项目特点

- 使用 ResNet18 作为基线，包含 8 个基本残差块。
- 针对 32×32 小图，将首层卷积改为 3×3、stride=1，并移除初始最大池化。
- 在每个 BasicBlock 的第二次卷积和 BN 之后、残差相加之前加入 SE 模块，`reduction=16`。
- 两组实验仅改变 `use_se`，其余训练设置保持一致。
- 按验证损失最低保存 `best.pth`，而不是直接使用最后一轮。
- 提供训练、断点续训、测试、绘图、数据检查和真实照片推理命令。

## 2. 实验结果

两组模型均训练 50 轮，在 10,000 张 CIFAR-10 测试图像上评价。

| 指标 | ResNet18 Baseline | SE-ResNet18 |
|---|---:|---:|
| 最佳权重轮次 | 47 | 50 |
| 测试损失 | 0.275130 | 0.261411 |
| Top-1 准确率 | 93.64% | 94.35% |
| Top-5 准确率 | 99.56% | 99.74% |
| 参数量 | 11,173,962 | 11,263,042 |
| 单张推理时间均值 | 2.2554 ms | 3.0170 ms |

SE 模型的 Top-1 准确率提高 0.71 个百分点，参数量增加 89,080（约 0.80%）。单张推理时间增加约 0.7616 ms。推理计时使用 batch size 1，预热 10 次、正式测量 100 次，并进行 CUDA 同步。

![训练曲线](results/training_curves.png)

分类别结果中，SE 在十类中的九类上提高，只有狗类下降 0.7 个百分点；猫类提升最大，由 85.7% 提高到 88.4%。完整数据见 `results/summary.json` 和两份 `per_class.csv`。

## 3. 场景应用

`scene_images` 包含从 Pixabay、Pexels 选取的 10 张公开真实照片，每个类别一张。这些照片不参与训练或调参。使用 SE 最佳权重进行推理后：

- Top-1 正确 8/10；
- Top-5 命中 10/10；
- 小汽车被误判为卡车，狗被误判为猫。

该结果只描述所选 10 张照片，不代表任意真实场景中的普遍准确率。逐图摘要见 `results/scene/inference_summary.csv`，照片来源见 `scene_images/图片来源与使用说明.txt`。

## 4. 环境安装

建议使用 Python 3.12 和支持 CUDA 的 PyTorch。本次实验环境为 Windows、PyTorch 2.15.0+cu132、torchvision 0.29.2、RTX 4080 Laptop GPU。

在 PyCharm 中打开本项目后，在下方 Terminal 执行：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
```

如果没有合适的 NVIDIA GPU，也可以安装 CPU 版 PyTorch，但训练速度会明显降低。可运行以下命令检查：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 5. 运行方法

以下命令均在项目根目录执行。

### 5.1 快速检查与数据准备

```bash
python main.py smoke
python main.py check-data
```

首次读取数据时会自动下载 CIFAR-10 到 `data/`。数据划分为 45,000 张训练、5,000 张验证和 10,000 张测试。

### 5.2 从头训练

```bash
python main.py train --model baseline --epochs 50
python main.py train --model se --epochs 50
```

默认 batch size 为 128，初始学习率为 0.1，使用 SGD、momentum 0.9、weight decay 5e-4、label smoothing 0.1 和 CosineAnnealingLR。

### 5.3 断点续训

```bash
python main.py train --model baseline --epochs 50 --resume runs/baseline/last.pth
python main.py train --model se --epochs 50 --resume runs/se/last.pth
```

`--epochs 50` 表示训练到总计第 50 轮，不是在已有轮次上再增加 50 轮。

### 5.4 测试与绘图

```bash
python main.py evaluate
python main.py plot
```

`evaluate` 需要 `runs/baseline/best.pth` 和 `runs/se/best.pth`；`plot` 只需要仓库中已有的两份训练日志。权重文件较大，未上传到仓库，运行训练后会自动重新生成。

### 5.5 真实照片推理

```bash
python main.py infer --input "scene_images"
```

默认读取 `runs/se/best.pth`，原始 Top-5 输出会保存到 `results/scene/inference_results.csv`。也可以传入一张图片的路径，或用 `--checkpoint` 指定其他权重。

## 6. 目录结构

```text
SE-ResNet18-CIFAR10/
├── main.py                         # 完整实验代码
├── requirements.txt
├── README.md
├── .gitignore
├── runs/
│   ├── baseline/config.json
│   ├── baseline/history.csv        # 正式50轮日志
│   ├── se/config.json
│   └── se/history.csv              # 正式50轮日志
├── results/
│   ├── summary.json                # 正式测试指标
│   ├── training_curves.png
│   ├── baseline_confusion_matrix.png
│   ├── se_confusion_matrix.png
│   ├── baseline_per_class.csv
│   ├── se_per_class.csv
│   └── scene/inference_summary.csv
└── scene_images/                   # 10张公开照片及来源说明
```

## 7. 不上传的内容

仓库不包含 CIFAR-10 数据集、`.venv`、`.idea`、`__pycache__`、`best.pth`、`last.pth` 或其他模型权重。这些数据与权重均可通过上述命令在本地重新生成。

## 参考资料

1. Krizhevsky A, Hinton G. Learning Multiple Layers of Features from Tiny Images. 2009.
2. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.
3. Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR, 2018.
