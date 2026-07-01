"""
实验三：使用PyTorch实现ResNet手写数字识别
==============================================
要求：
  1. 层数不限，建议50层
  2. 数据集使用MNIST
  3. 完成训练和预测
  4. 分析残差连接的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# =============================================
# 1. ResNet组件
# =============================================

class BasicBlock(nn.Module):
    """ResNet基础残差块
    
    适用于ResNet18/34的BasicBlock，包含两个3x3卷积
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 关键：残差连接
        out = F.relu(out, inplace=True)
        
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块
    
    适用于ResNet50/101/152，包含1x1-3x3-1x1卷积
    扩展因子为4，减少参数量
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1卷积升维（expansion倍）
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 残差连接
        out = F.relu(out, inplace=True)
        
        return out


class ResNet(nn.Module):
    """ResNet主网络
    
    支持多种层数配置：ResNet-18, ResNet-34, ResNet-50, ResNet-101
    """
    
    def __init__(self, block, layers, num_classes=10, input_channels=1):
        """
        Args:
            block: 残差块类型（BasicBlock 或 Bottleneck）
            layers: 每个阶段的残差块数量列表，如 [3, 4, 6, 3] 对应 ResNet-50
            num_classes: 分类数
            input_channels: 输入通道数（MNIST为1，彩色图为3）
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层（适配MNIST的小尺寸输入，使用较小的kernel和stride）
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # MNIST图片较小（28x28），不使用MaxPool避免过度降采样
        
        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建一个残差层"""
        downsample = None
        
        # 当维度不匹配时需要downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # 第一个block可能需要stride和downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 后续block
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# 预定义的ResNet变体
def resnet18(num_classes=10, input_channels=1):
    """ResNet-18: [2,2,2,2] 层BasicBlock"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


def resnet34(num_classes=10, input_channels=1):
    """ResNet-34: [3,4,6,3] 层BasicBlock"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)


def resnet50(num_classes=10, input_channels=1):
    """ResNet-50: [3,4,6,3] 层Bottleneck（推荐使用）"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)


def resnet101(num_classes=10, input_channels=1):
    """ResNet-101: [3,4,23,3] 层Bottleneck"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_channels)


# =============================================
# 2. PlainNet（普通卷积网络，用于对比残差连接效果）
# =============================================

class PlainBlock(nn.Module):
    """普通卷积块（无残差连接）"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out, inplace=True)
        return out  # 注意：没有残差连接


class PlainNet(nn.Module):
    """无残差连接的普通深层网络（用于对比）"""
    
    def __init__(self, block, layers, num_classes=10, input_channels=1):
        super(PlainNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================
# 3. 训练和评估函数
# =============================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, device, model_name="resnet", patience=15):
    """完整训练流程（带早停和学习率调度）"""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    save_dir = "./exp3_resnet/results"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{save_dir}/{model_name}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发于 epoch {epoch+1}，最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch})")
                break
    
    model.load_state_dict(torch.load(f"{save_dir}/{model_name}_best.pth"))
    return model, history, best_val_acc


def plot_training_history(history, model_name, save_path=None):
    """绘制训练历史曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=1.5)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(model, dataloader, device, save_path=None):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('ResNet-50 Confusion Matrix on MNIST')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================
# 4. 主实验：ResNet-50 vs PlainNet 对比
# =============================================

def compare_resnet_vs_plain():
    """ResNet和PlainNet对比实验"""
    print("=" * 70)
    print("实验3：ResNet vs PlainNet 残差连接效果对比")
    print("=" * 70)
    
    # 数据准备
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    models_to_train = [
        ("ResNet-18", resnet18(), 0.001, 50),
        ("ResNet-50", resnet50(), 0.001, 50),
    ]
    
    results = {}
    histories = {}
    
    for name, model, lr, epochs in models_to_train:
        print(f"\n{'='*50}")
        print(f"训练: {name} (参数量: {count_parameters(model):,})")
        print(f"{'='*50}")
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        model, history, best_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            epochs, device, model_name=name
        )
        
        histories[name] = history
        results[name] = best_acc
        
        # 测试集评估
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        results[f"{name}_test"] = test_acc
        print(f"  {name} - 测试准确率: {test_acc:.4f}")
        
        # 绘制训练曲线
        plot_training_history(history, name,
                             save_path=f"./exp3_resnet/results/{name}_history.png")
    
    # 绘制对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['blue', 'red']
    
    for idx, (name, history) in enumerate(histories.items()):
        epochs_range = range(1, len(history['val_loss']) + 1)
        axes[0].plot(epochs_range, history['val_loss'], color=colors[idx], label=name, linewidth=1.5)
        axes[1].plot(epochs_range, history['val_acc'], color=colors[idx], label=name, linewidth=1.5)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('ResNet Variants - Validation Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('ResNet Variants - Validation Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./exp3_resnet/results/resnet_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 使用ResNet-50绘制混淆矩阵
    best_resnet50 = resnet50().to(device)
    best_resnet50.load_state_dict(
        torch.load("./exp3_resnet/results/ResNet-50_best.pth")
    )
    plot_confusion_matrix(best_resnet50, test_loader, device,
                         save_path="./exp3_resnet/results/resnet50_confusion.png")
    
    # 展示残差连接的效果（通过可视化梯度和中间层激活值）
    print("\n残差连接分析:")
    print("  ResNet通过跳跃连接（Skip Connection）解决了深层网络中的梯度消失问题")
    print("  允许网络在层数加深的同时保持甚至提升性能")
    
    return results


# =============================================
# 主程序
# =============================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  深度学习课程实验三：ResNet手写数字识别")
    print("#" * 70)
    
    start_time = time.time()
    
    # ResNet vs PlainNet 对比实验
    results = compare_resnet_vs_plain()
    
    total_time = time.time() - start_time
    print(f"\n实验三总耗时: {total_time:.2f}s")
    print("实验三全部完成！")
    
    print("\n" + "=" * 70)
    print("实验结果汇总:")
    print("=" * 70)
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
    
    print("\n参数量对比:")
    for name in ["ResNet-18", "ResNet-50"]:
        if "18" in name:
            model = resnet18()
        else:
            model = resnet50()
        params = count_parameters(model)
        print(f"  {name}: {params:,} 参数")
