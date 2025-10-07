import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 首先设置数据变换
from torchvision import transforms


image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),  
     # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])
## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 256
num_workers = 4   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
lr = 1e-4
epochs = 20

## 读取方式二：读入csv格式的数据，自行构建Dataset类
# csv数据下载链接：https://www.kaggle.com/zalando-research/fashionmnist
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 使用相对路径读取数据（相对于项目根目录）
from pathlib import Path
project_root = Path(__file__).parent.parent
train_df = pd.read_csv(project_root / "data" / "fashion-mnist_train.csv")
test_df = pd.read_csv(project_root / "data" / "fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#搭建CNN

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

# 自动选择最佳设备：MPS (Mac GPU) > CUDA (NVIDIA GPU) > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ 使用 Apple Silicon GPU (MPS) 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ 使用 NVIDIA GPU (CUDA) 加速")
else:
    device = torch.device("cpu")
    print("✓ 使用 CPU")

model = Net()
model = model.to(device)

#损失函数
criterion = nn.CrossEntropyLoss()
#优化器 (使用顶部定义的学习率)
optimizer = optim.Adam(model.parameters(), lr=lr)

#训练
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
        # 每10个batch显示一次进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss = train_loss/len(train_loader.dataset)
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')
    return train_loss
    
    
#验证
def val(epoch):       
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}, Accuracy: {acc:.4f} ({acc*100:.2f}%)\n')
    return val_loss, acc
    
#启动训练
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"开始训练 Fashion-MNIST 分类模型")
    print(f"{'='*60}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {epochs}")
    print(f"设备: {device}")
    print(f"{'='*60}\n")
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs+1):
        train_loss = train(epoch)
        val_loss, acc = val(epoch)
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ 最佳模型已保存！准确率: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
    
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%) at Epoch {best_epoch}")
    print(f"{'='*60}")