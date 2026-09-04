# Oxford-IIIT Pet 37 类宠物图像分类

本项目使用迁移学习完成 Oxford-IIIT Pet 37 类宠物品种分类，并在 ResNet-18 基线模型上进行 Label Smoothing 单变量消融实验。

## 1. 实验任务

- 数据集：Oxford-IIIT Pet，37 个类别、7,349 张图像。
- 数据划分：合并数据集官方 trainval 与 test 后，按类别分层划分为训练/验证/测试 = 70%/15%/15%。
- 随机种子：42。
- 划分结果：训练集 5,144 张；验证集 1,102 张；测试集 1,103 张。
- 模型：ImageNet 预训练 ResNet-18，最后全连接层替换为 37 类输出。
- 训练：AdamW，学习率 1e-4，weight decay 1e-4，batch size 32，训练 15 epoch。
- 实际运行环境：Kaggle Tesla T4 GPU；PyTorch 2.10.0+cu128；torchvision 0.25.0+cu128。

注意：本作业明确要求以 7,349 张图像完成 70/15/15 划分，因此这里没有直接沿用数据集自带的官方 train/test 划分。

## 2. 项目结构

    pet-classifier/
    ├── data/
    │   └── dataset.py       # 下载、增强、分层划分和 DataLoader
    ├── models/
    │   └── model.py         # 预训练 ResNet-18
    ├── utils/
    │   ├── metrics.py       # Top-1、Top-5、Macro-F1 与混淆矩阵
    │   └── gradcam.py       # Grad-CAM 可解释性工具
    ├── train.py             # 训练入口
    ├── evaluate.py          # 测试和混淆矩阵入口
    ├── requirements.txt
    └── README.md

原始数据和模型权重不会上传 GitHub；它们在 .gitignore 中被忽略。训练产生的
TensorBoard event 文件应作为独立材料一并提交（本实验的实际日志已在 Kaggle 备份包的
runs 文件夹中）。如需要放在仓库中，可将该文件夹复制为 logs 后再提交。

## 3. 环境与运行

在 Kaggle 或已配置 CUDA 的 Python 环境中安装依赖：

    pip install -r requirements.txt

下面命令均在本项目根目录运行。

训练基线模型：

    python train.py --experiment-name baseline --label_smoothing 0.0

只改变损失函数的 label smoothing 系数，运行单变量消融：

    python train.py --experiment-name label_smoothing --label_smoothing 0.1

使用已保存的最佳模型在测试集评估并保存混淆矩阵：

    python evaluate.py --checkpoint outputs/checkpoints/label_smoothing_best_model.pth --save-confusion-matrix

查看训练日志：

    tensorboard --logdir outputs/runs

在 Kaggle 发生 DataLoader 多进程报错时，保留默认参数 num_workers=0 即可。这只改变数据读取方式，不改变数据集、模型或实验设置。

## 4. 当前实验结果

| Experiment | Best epoch | Validation Top-1 | Test Top-1 | Test Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 11 | 92.4682% | 90.2992% | 99.1840% | 0.9026 |
| Label Smoothing (0.1) | 7 | 93.1942% | 92.2937% | 98.7307% | 0.9225 |

相对基线，Label Smoothing 的测试 Top-1 提升 1.9945 个百分点，Macro-F1 提升 0.0199；Top-5 小幅下降 0.4533 个百分点。因此，在本实验中它提升了整体分类性能与类别间的均衡性。

## 5. 结果解读

- 训练曲线显示训练准确率继续上升而验证准确率较早进入平台，说明基线后期存在轻微过拟合。
- 两个实验的损失函数定义不同，不能直接以绝对 loss 大小横向判定优劣；应以同一测试集上的 Top-1、Top-5 与 Macro-F1 为主。
- 主要混淆包括 Egyptian Mau → Bengal（5 次）、American Pit Bull Terrier → Staffordshire Bull Terrier（5 次）、Basset Hound → Beagle（3 次）。这些品种有相似的毛色、头部或体型特征。
- Grad-CAM 正确案例主要关注 Yorkshire Terrier 的脸部、耳朵和毛发；错误案例主要关注 American Pit Bull Terrier 的头部和身体，说明模型关注到了动物本身，但相近品种的细粒度差异仍难以区分。

## 6. AI 使用说明与调试记录

本项目在学习和开发过程中使用 AI 助手辅助解释深度学习术语、梳理代码结构、生成初版模板与排查报错。实验由作者本人在 Kaggle GPU 环境中执行，所有指标、图表与结果均来自实际运行输出，并经作者核对。

一次具体调试记录：Codex 给出的初版 DataLoader 使用 num_workers=2；在 Kaggle Notebook 会话重启后，它出现了 AssertionError: can only test a child process。将 num_workers 改为 0 后，数据加载恢复稳定，并确认图片 batch 形状仍为 [32, 3, 224, 224]、标签 batch 形状仍为 [32]。这不是模型或数据划分的修改。

## 7. 提交时的文件边界

- 上传 GitHub：本项目源码、README、requirements.txt，以及可以公开展示的少量结果图。
- 不上传 GitHub：原始数据集、.pth 权重、236 MB 的完整备份 ZIP。
- 单独随作业提交：2–3 页 PDF 技术报告，以及 TensorBoard 的 runs 日志文件夹（或其压缩包）。
