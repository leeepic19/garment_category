[README.md](https://github.com/user-attachments/files/22742240/README.md)
# 🎯 Fashion-MNIST 服装图片分类项目

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

使用自定义 CNN 实现 Fashion-MNIST 数据集的十分类任务

[English](#english) | [中文](#chinese)

</div>

---

## <a name="chinese"></a>📖 项目简介

本项目使用 **PyTorch** 从零手动搭建卷积神经网络（CNN），实现对 Fashion-MNIST 服装数据集的图像分类。项目包含完整的数据加载、模型训练、验证和可视化流程，适合深度学习初学者学习和实践。

### ✨ 主要特性

- 🏗️ **手动搭建 CNN**：自定义 2 层卷积 + 2 层全连接网络
- 📊 **自定义 Dataset**：从 CSV 格式读取数据，手写数据加载类
- 🚀 **跨平台加速**：自动检测并使用 MPS (Mac) / CUDA (NVIDIA) / CPU
- 📈 **训练监控**：实时显示训练进度、损失和准确率
- 💾 **模型保存**：自动保存最佳验证准确率的模型
- 🎨 **数据可视化**：展示样本图片及类别标签

---

## 🎯 数据集介绍

**Fashion-MNIST** 是一个经典的图像分类数据集，由 Zalando 公司创建，包含 10 类服装图片：

| 标签 | 类别名称 | 示例 |
|------|---------|------|
| 0 | T-shirt/top (T恤) | 👕 |
| 1 | Trouser (裤子) | 👖 |
| 2 | Pullover (套衫) | 🧥 |
| 3 | Dress (连衣裙) | 👗 |
| 4 | Coat (外套) | 🧥 |
| 5 | Sandal (凉鞋) | 👡 |
| 6 | Shirt (衬衫) | 👔 |
| 7 | Sneaker (运动鞋) | 👟 |
| 8 | Bag (包) | 👜 |
| 9 | Ankle boot (短靴) | 👢 |

**数据集规模**:
- 训练集: 60,000 张图片
- 测试集: 10,000 张图片
- 图片尺寸: 28×28 像素（灰度图）

---

## 🏗️ 项目结构

```
服装十分类/
├── source/                      # 源代码目录
│   ├── main.py                 # 主训练脚本 ⭐
│   ├── test_loader.py          # 数据可视化工具
│   └── downloadData.py         # 数据下载脚本
├── data/                        # 数据目录
│   ├── fashion-mnist_train.csv # 训练集CSV
│   └── fashion-mnist_test.csv  # 测试集CSV
├── best_model.pth              # 最佳模型权重
├── data_visualization.png      # 数据样本可视化
├── requirements.txt            # 依赖清单
└── README.md                   # 项目说明
```

---

## 🛠️ 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**:
- `torch==2.8.0` - PyTorch 深度学习框架
- `torchvision==0.23.0` - 图像处理工具
- `numpy==2.3.3` - 数值计算
- `pandas==2.3.3` - 数据处理
- `matplotlib==3.10.6` - 可视化

### 4. 下载数据集

从 [Kaggle Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist) 下载 CSV 格式数据，放置到 `data/` 目录：

```
data/
├── fashion-mnist_train.csv
└── fashion-mnist_test.csv
```

---

## 🚀 快速开始

### 训练模型

```bash
python source/main.py
```

**训练过程示例**:
```
============================================================
开始训练 Fashion-MNIST 分类模型
============================================================
训练集大小: 60000
测试集大小: 10000
Batch size: 256
学习率: 0.0001
训练轮数: 20
设备: mps
============================================================

✓ 使用 Apple Silicon GPU (MPS) 加速

Epoch: 1 [2304/60000 (4%)]      Loss: 2.253430
Epoch: 1 [4864/60000 (8%)]      Loss: 2.161488
...
Epoch: 1        Training Loss: 1.111603
Epoch: 1        Validation Loss: 0.625431, Accuracy: 0.7823 (78.23%)

✓ 最佳模型已保存！准确率: 0.7823 (78.23%)
```

### 数据可视化

```bash
python source/test_loader.py
```

将生成 `data_visualization.png`，展示 16 张样本图片：

![数据可视化示例](data_visualization.png)

---

## 🧠 模型架构

### CNN 网络结构

```
输入: [batch, 1, 28, 28]
    ↓
卷积层1: Conv2d(1→32, 5×5) + ReLU + MaxPool(2×2) + Dropout(0.3)
    ↓ [batch, 32, 12, 12]
卷积层2: Conv2d(32→64, 5×5) + ReLU + MaxPool(2×2) + Dropout(0.3)
    ↓ [batch, 64, 4, 4]
展平: Flatten
    ↓ [batch, 1024]
全连接1: Linear(1024→512) + ReLU
    ↓ [batch, 512]
全连接2: Linear(512→10)
    ↓ [batch, 10]
输出: 10类别的Logits
```

### 模型参数

| 指标 | 数值 |
|------|------|
| 总参数量 | 582,026 |
| 模型大小 | ~2.2 MB |
| 卷积层参数 | 52,096 |
| 全连接层参数 | 529,930 |

### 超参数配置

```python
batch_size = 256        # 批次大小
lr = 1e-4               # 学习率
epochs = 20             # 训练轮数
optimizer = Adam        # 优化器
loss = CrossEntropy     # 损失函数
dropout = 0.3           # Dropout比例
```

---

## 📊 性能指标

### 预期结果

| 指标 | 数值 |
|------|------|
| 测试准确率 | 85-90% |
| 训练时间 (MPS) | ~3-5 分钟 |
| 训练时间 (CPU) | ~15-20 分钟 |
| 推理速度 | < 1秒/10,000样本 |

### 设备支持

- ✅ **Apple Silicon (M1/M2/M3)**: 使用 MPS 加速
- ✅ **NVIDIA GPU**: 使用 CUDA 加速
- ✅ **CPU**: 纯 CPU 模式（较慢但可用）

---

## 📂 核心代码解析

### 1. 自定义 Dataset 类

```python
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.images = df.iloc[:,1:].values.astype(np.uint8)  # 像素数据
        self.labels = df.iloc[:, 0].values                    # 标签
        self.transform = transform
        
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
```

### 2. CNN 模型定义

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        return self.fc(x)
```

### 3. 跨平台设备管理

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")           # Mac GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")          # NVIDIA GPU
else:
    device = torch.device("cpu")           # CPU
```

---

## 🎨 可视化结果

运行 `test_loader.py` 后将生成数据可视化图片，展示：
- 4×4 网格的 16 张样本图片
- 每张图片对应的类别标签
- 清晰的灰度图像显示

---

## 🔧 自定义配置

### 修改超参数

编辑 `source/main.py` 中的配置：

```python
batch_size = 256    # 调整批次大小
lr = 1e-4           # 调整学习率
epochs = 20         # 调整训练轮数
num_workers = 4     # 调整数据加载线程数（Windows设为0）
```

### 调整模型结构

修改 `Net` 类中的卷积层或全连接层：

```python
# 增加卷积层深度
nn.Conv2d(64, 128, 3)

# 调整全连接层大小
nn.Linear(1024, 256)
```

---

## 📝 常见问题

### Q1: Mac 上报错 "CUDA not available"
**A**: 项目已自动适配，Mac 会使用 MPS 加速，无需 CUDA。

### Q2: Windows 上多进程报错
**A**: 将 `num_workers` 设为 0：
```python
num_workers = 0  # Windows 用户
```

### Q3: 如何加载已保存的模型？
**A**: 使用以下代码：
```python
model = Net()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

### Q4: 如何提升准确率？
**A**: 可尝试：
- 增加训练轮数
- 添加数据增强（RandomRotation, RandomHorizontalFlip）
- 使用学习率调度器
- 尝试更深的网络结构

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) - Zalando Research
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Kaggle](https://www.kaggle.com/) - 数据集托管平台

---

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: your.email@example.com
- 🐙 GitHub: [@your-username](https://github.com/your-username)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

Made with ❤️ by [Your Name]

</div>

---

## <a name="english"></a>📖 English Version

### Project Overview

A Fashion-MNIST image classification project built from scratch using PyTorch and custom CNN architecture.

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model
python source/main.py
```

### Key Features

- 🏗️ Custom CNN architecture (2 conv + 2 fc layers)
- 📊 Manual Dataset class for CSV data loading
- 🚀 Cross-platform acceleration (MPS/CUDA/CPU)
- 📈 Real-time training monitoring
- 💾 Automatic best model saving
- 🎨 Data visualization tool

### Model Architecture

```
Input (28×28×1)
    ↓
Conv2d(1→32, 5×5) → ReLU → MaxPool → Dropout(0.3)
    ↓
Conv2d(32→64, 5×5) → ReLU → MaxPool → Dropout(0.3)
    ↓
Linear(1024→512) → ReLU → Linear(512→10)
    ↓
Output (10 classes)
```

### Expected Performance

- **Test Accuracy**: 85-90%
- **Training Time (MPS)**: ~3-5 minutes
- **Parameters**: 582,026
- **Model Size**: ~2.2 MB

### License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Star ⭐ this repository if you find it helpful!**

</div>
