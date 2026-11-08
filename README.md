# 机器学习课程设计 · 四项视觉任务

本仓库为《机器学习》课程设计四个任务的完整实现代码，覆盖图像分类、人脸表情识别、关键点（中心凹）定位与医学图像分割四类典型视觉问题。四个任务均在 Kaggle Notebook（Tesla T4 ×2）上完成训练与推理，提交文件均由脚本自动生成。

## 一、任务与结果概览

| 任务 | 任务类型 | 数据规模 | 主要方法 | 评价指标 | 本地验证结果 |
| --- | --- | --- | --- | --- | --- |
| 任务一 | 作物幼苗 5 类图像分类 | 训练 500 张 / 测试 378 张 | EfficientNet-B0 迁移学习 + ResNet18 五折集成 + 4 视角 TTA + 概率加权融合 | Macro-F1 | 单模型验证 Macro-F1 **0.9395**；五折平均 0.8916 |
| 任务二 | 人脸表情 6 类识别 | 测试 7065 张（纯推理） | 预训练 ViT 迁移 + 7→6 类标签空间重映射 + 水平翻转 TTA | 分类准确率 | 无需训练，输出 6 类预测分布 |
| 任务三 | 眼底黄斑中心凹定位 | 训练 80 张 / 测试 20 张 | timm 骨干 + 1/8 分辨率热图头 + soft-argmax 坐标回归；五折、几何一致增强、双视图 TTA | RMSE（像素） | 五折平均 **18.0 px**，均值基线 110.5 px |
| 任务四 | 眼底血管分割 | 训练 20 张 / 测试 20 张 | Green+CLAHE 预处理、256 patch 训练、弹性形变、timm 编码器 UNet、BCE+Dice+clDice、2 种子 × 5 折集成、多尺度 TTA、OOF 阈值搜索 | Dice | 折外（OOF）Dice **0.8029**（阈值 0.20） |

## 二、环境依赖

- Python 3.10+，PyTorch 2.x，CUDA 11.8/12.x
- 主要第三方库见 `requirements.txt`：`torch`、`torchvision`、`timm`、`transformers`、`scikit-learn`、`pandas`、`numpy`、`Pillow`、`tqdm`
- 训练在 Kaggle 提供的 Tesla T4 ×2 环境中完成，脚本内已启用混合精度（AMP）

```bash
pip install -r requirements.txt
```

## 三、目录结构

```
ml-course-design/
├── task1_seedling_classification.py    # 任务一：作物幼苗图像分类
├── task2_fer_emotion_recognition.py    # 任务二：人脸表情识别
├── task3_fovea_localization.py         # 任务三：黄斑中心凹定位
├── task4_vessel_segmentation.py        # 任务四：眼底血管分割
├── requirements.txt
└── README.md
```

## 四、运行方式

四个脚本均按「单文件、自上而下顺序执行」组织，在 Kaggle Notebook 中新建 Notebook 后整段粘贴运行即可。数据目录与提交模板路径以常量形式写在每个脚本开头：

```python
BASE      = '/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2'
SUB_CSV   = '/kaggle/input/competitions/2026task1/submission-for-task2.csv'
```

若在本地或其他平台运行，只需将上述路径常量改为实际数据路径，其余逻辑无需改动。任务三、任务四的数据目录、掩膜目录与提交表列名均做了运行时探测，会自动识别标注文件、坐标单位与提交格式。

各脚本的输出：

- 任务一：`/kaggle/working/submission.csv`（378 行，`ID, Category`）
- 任务二：`/kaggle/working/submission.csv`（7065 行，`ID, Emotion`）
- 任务三：`/kaggle/working/submission.csv`（40 行，`ImageID, value`）
- 任务四：`/kaggle/working/submission.csv`（主，阈值 0.20）与 `submission_alt.csv`（备用，阈值 0.55）

## 五、关键实验设置

| 配置项 | 任务一 | 任务三 | 任务四 |
| --- | --- | --- | --- |
| 输入尺寸 | 224 × 224 | 448 × 448 | 256 × 256 patch（滑窗推理） |
| 骨干网络 | EfficientNet-B0 / ResNet18 | EfficientNet-B0 / ResNet18（逐折交替） | EfficientNet-B0 / ResNet18（逐折交替） |
| 损失函数 | CrossEntropy（label smoothing 0.1） | Smooth L1 + 0.5 × 热图 BCE | BCE + Dice + 0.5 × clDice |
| 优化器 | AdamW（lr 1e-3, wd 1e-4） | AdamW（lr 2e-4, wd 0.05） | AdamW（lr 3e-4, wd 1e-4） |
| 学习率调度 | CosineAnnealingLR | Warmup + 余弦退火（逐步更新） | Warmup + 余弦退火（逐步更新） |
| 验证策略 | 分层 8:2 留出 + 5 折交叉验证 | 5 折交叉验证 | 5 折 × 2 种子交叉验证 |

## 六、说明

- 本仓库为课程设计最终提交版本，代码按提交规范整理为可直接运行的脚本（去除冗余注释与调试输出），逻辑与实验运行时完全一致。
- 任务二为迁移推理方案：使用 HuggingFace 上的预训练表情模型（`abhilash88/face-emotion-detection`），并按竞赛要求剔除 Disgust 类，将 7 类输出映射为 6 类。
- 任务四的 RLE 编码严格对齐官方 `segmentation_to_csv.py`：1-based、C 序展平、在 512 × 512 尺度上编码。

## 七、参考

1. Tan M, Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
2. He K, et al. Deep Residual Learning for Image Recognition. CVPR, 2016.
3. Dosovitskiy A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.
4. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
5. Shit S, et al. clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR, 2021.
