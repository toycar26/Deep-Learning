"""
实验一：使用numpy编写多层感知机（MLP）
==============================================
要求：
  1. 只能使用numpy，不能直接使用pytorch中的linear、sigmoid等函数
  2. 完成输入层、隐含层和输出层
  3. 编写损失函数以及激活函数
  4. 编写训练过程、预测过程
  5. 分类任务，数据集自行选择（使用sklearn的make_moons生成非线性二分类数据）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time

# 设置随机种子以保证结果可复现
np.random.seed(42)

# =============================================
# 1. 激活函数及其导数
# =============================================

class Sigmoid:
    """Sigmoid激活函数: f(x) = 1 / (1 + e^{-x})"""
    @staticmethod
    def forward(x):
        # 数值稳定性处理：裁剪极端值
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def backward(x):
        """输入为前向传播后的激活值"""
        s = Sigmoid.forward(x)
        return s * (1 - s)


class Tanh:
    """双曲正切激活函数: f(x) = (e^x - e^{-x}) / (e^x + e^{-x})"""
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(x):
        """输入为前向传播后的激活值"""
        return 1.0 - np.power(np.tanh(x), 2)


class ReLU:
    """ReLU激活函数: f(x) = max(0, x)"""
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        """输入为前向传播的原始值（非激活值）"""
        return (x > 0).astype(np.float64)


class Softmax:
    """Softmax激活函数（用于多分类输出层）"""
    @staticmethod
    def forward(x):
        # 数值稳定性：减去每行最大值
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def backward(x, y_true):
        """Softmax + CrossEntropy的联合导数"""
        return x - y_true


# =============================================
# 2. 损失函数
# =============================================

class CrossEntropyLoss:
    """交叉熵损失（与Softmax配合使用，已经合并导数）"""
    @staticmethod
    def forward(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        n = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / n
        return loss
    
    @staticmethod
    def backward(y_pred, y_true):
        """这个函数不会直接使用，由Softmax.backward处理"""
        return (y_pred - y_true) / y_true.shape[0]


class BCELoss:
    """二分类交叉熵损失（与Sigmoid配合使用）"""
    @staticmethod
    def forward(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def backward(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


class MSELoss:
    """均方误差损失"""
    @staticmethod
    def forward(y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))
    
    @staticmethod
    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]


# =============================================
# 3. MLP网络层
# =============================================

class Linear:
    """全连接层: y = Wx + b"""
    def __init__(self, in_features, out_features, init_type='he'):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            init_type: 权重初始化方式 ('he', 'xavier', 'random')
        """
        self.in_features = in_features
        self.out_features = out_features
        
        if init_type == 'he':
            # He初始化（适用于ReLU）
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        elif init_type == 'xavier':
            # Xavier初始化（适用于tanh/sigmoid）
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01
        
        self.b = np.zeros((1, out_features))
        
        # 缓存前向传播的中间值（用于反向传播）
        self.x = None
    
    def forward(self, x):
        """前向传播"""
        self.x = x  # 缓存输入
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout, learning_rate):
        """反向传播 + 参数更新"""
        # dout: 上游梯度，形状 (batch_size, out_features)
        batch_size = self.x.shape[0]
        
        # 计算梯度
        dW = np.dot(self.x.T, dout) / batch_size
        db = np.sum(dout, axis=0, keepdims=True) / batch_size
        dx = np.dot(dout, self.W.T)  # 传递给前一层的梯度
        
        # 参数更新（SGD）
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dx


# =============================================
# 4. MLP模型
# =============================================

class MLP:
    """多层感知机模型"""
    
    def __init__(self, layer_dims, activation='relu', output_activation='sigmoid',
                 loss='bce', learning_rate=0.01):
        """
        Args:
            layer_dims: 各层维度列表，如 [2, 16, 8, 1]
            activation: 隐含层激活函数 ('relu', 'sigmoid', 'tanh')
            output_activation: 输出层激活函数 ('sigmoid', 'softmax', 'linear')
            loss: 损失函数类型 ('bce', 'mse', 'cross_entropy')
            learning_rate: 学习率
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_name = loss
        
        # 构建网络层
        self.layers = []
        for i in range(len(layer_dims) - 1):
            # 隐含层用He初始化（适合ReLU），输出层用Xavier
            init = 'he' if i < len(layer_dims) - 2 else 'xavier'
            self.layers.append(Linear(layer_dims[i], layer_dims[i+1], init_type=init))
        
        # 选择激活函数
        activation_map = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}
        self.hidden_activation = activation_map[activation]
        
        # 选择输出层激活函数
        output_activation_map = {'sigmoid': Sigmoid, 'softmax': Softmax, 'linear': None}
        self.output_activation = output_activation_map[output_activation]
        
        # 选择损失函数
        loss_map = {'bce': BCELoss, 'mse': MSELoss, 'cross_entropy': CrossEntropyLoss}
        self.loss_fn = loss_map[loss]
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def forward(self, x):
        """前向传播，返回各层的中间cache"""
        cache = {'z': [], 'a': [x]}  # z: 线性输出, a: 激活输出
        
        a = x
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            cache['z'].append(z)
            
            # 最后一层使用输出激活函数
            if i == len(self.layers) - 1:
                if self.output_activation is not None:
                    a = self.output_activation.forward(z)
                else:
                    a = z
            else:
                a = self.hidden_activation.forward(z)
            
            cache['a'].append(a)
        
        # 保存激活值用于反向传播（修正：需要保存线性输出值用于后向传播）
        # 对于使用backward的中间层，缓存原始z值
        cache['hidden_z'] = [z for z in cache['z'][:-1]]  # 隐含层的线性输出
        cache['output_z'] = cache['z'][-1]  # 输出层的线性输出
        
        return a, cache
    
    def backward(self, y_pred, y_true, cache, learning_rate):
        """反向传播"""
        batch_size = y_true.shape[0]
        
        # 输出层梯度
        if self.output_activation_name == 'sigmoid':
            # Sigmoid + BCE: 手动计算联合梯度
            dout = (y_pred - y_true) / batch_size
        elif self.output_activation_name == 'softmax':
            # Softmax + CrossEntropy: 联合梯度
            dout = Softmax.backward(y_pred, y_true)
        else:
            # Linear + MSE
            dout = self.loss_fn.backward(y_pred, y_true)
        
        # 从后向前逐层反向传播
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                # 输出层
                dout = self.layers[i].backward(dout, learning_rate)
            else:
                # 隐含层：先通过激活函数的导数
                z = cache['hidden_z'][i]
                if self.activation_name == 'relu':
                    dout = dout * ReLU.backward(z)
                elif self.activation_name == 'sigmoid':
                    dout = dout * Sigmoid.backward(z)
                elif self.activation_name == 'tanh':
                    dout = dout * Tanh.backward(z)
                
                # 再通过线性层的反向传播
                dout = self.layers[i].backward(dout, learning_rate)
    
    def compute_loss(self, y_pred, y_true):
        """计算损失"""
        return self.loss_fn.forward(y_pred, y_true)
    
    def compute_accuracy(self, x, y):
        """计算准确率"""
        y_pred, _ = self.forward(x)
        if y_pred.shape[1] == 1:
            # 二分类
            pred_classes = (y_pred >= 0.5).astype(int).flatten()
            true_classes = y.flatten()
        else:
            # 多分类
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
        return np.mean(pred_classes == true_classes)
    
    def fit(self, x_train, y_train, x_val=None, y_val=None,
            epochs=1000, batch_size=32, verbose=True):
        """训练模型"""
        n_samples = x_train.shape[0]
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            # Mini-batch SGD
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # 前向传播
                y_pred, cache = self.forward(x_batch)
                
                # 计算损失
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss * (end - start)
                
                # 反向传播
                self.backward(y_pred, y_batch, cache, self.learning_rate)
            
            epoch_loss /= n_samples
            self.train_losses.append(epoch_loss)
            
            # 计算训练准确率
            train_acc = self.compute_accuracy(x_train, y_train)
            self.train_accs.append(train_acc)
            
            # 验证集评估
            if x_val is not None and y_val is not None:
                y_val_pred, _ = self.forward(x_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_acc = self.compute_accuracy(x_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # 保存最佳模型参数
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._best_params = [(layer.W.copy(), layer.b.copy()) for layer in self.layers]
                
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            elif verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f}")
    
    def predict(self, x):
        """预测"""
        y_pred, _ = self.forward(x)
        if y_pred.shape[1] == 1:
            return (y_pred >= 0.5).astype(int).flatten()
        else:
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, x):
        """返回概率预测"""
        y_pred, _ = self.forward(x)
        return y_pred
    
    def restore_best(self):
        """恢复最佳模型参数"""
        if hasattr(self, '_best_params'):
            for i, layer in enumerate(self.layers):
                layer.W, layer.b = self._best_params[i]


# =============================================
# 5. 数据准备与实验
# =============================================

def plot_decision_boundary(model, x, y, title="Decision Boundary", save_path=None):
    """绘制决策边界"""
    h = 0.01
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu, s=50)
    plt.title(title, fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(model, save_path=None):
    """绘制训练历史曲线"""
    epochs = range(1, len(model.train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(epochs, model.train_losses, 'b-', label='Train Loss', linewidth=1.5)
    if model.val_losses:
        axes[0].plot(epochs, model.val_losses, 'r-', label='Val Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, model.train_accs, 'b-', label='Train Accuracy', linewidth=1.5)
    if model.val_accs:
        axes[1].plot(epochs, model.val_accs, 'r-', label='Val Accuracy', linewidth=1.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def experiment_1_basic():
    """实验1：基本二分类任务（make_moons数据集）"""
    print("=" * 70)
    print("实验一：MLP在非线性二分类数据集上的表现")
    print("=" * 70)
    
    # 生成数据
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)  # 转为列向量
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
    print(f"输入特征维度: {X_train.shape[1]}")
    
    # 创建MLP模型: 2 -> 16 -> 8 -> 1
    model = MLP(
        layer_dims=[2, 16, 8, 1],
        activation='relu',
        output_activation='sigmoid',
        loss='bce',
        learning_rate=0.1
    )
    
    print(f"网络结构: {model.layer_dims}")
    print(f"激活函数: {model.activation_name}")
    print(f"输出激活: {model.output_activation_name}")
    print(f"损失函数: {model.loss_name}")
    print(f"学习率: {model.learning_rate}")
    print()
    
    # 训练
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val, epochs=2000, batch_size=64, verbose=True)
    train_time = time.time() - start_time
    
    # 恢复最佳模型
    model.restore_best()
    
    # 评估
    train_acc = model.compute_accuracy(X_train, y_train)
    val_acc = model.compute_accuracy(X_val, y_val)
    
    print(f"\n训练时间: {train_time:.2f}s")
    print(f"最终训练准确率: {train_acc:.4f}")
    print(f"最终验证准确率: {val_acc:.4f}")
    
    # 绘制决策边界和训练曲线
    os.makedirs("./exp1_mlp/results", exist_ok=True)
    plot_decision_boundary(model, X, y, 
                           title=f"MLP Decision Boundary (Val Acc: {val_acc:.4f})",
                           save_path="./exp1_mlp/results/decision_boundary.png")
    plot_training_history(model,
                          save_path="./exp1_mlp/results/training_history.png")
    
    print("\n结果已保存至 exp1_mlp/results/ 目录")
    return model, val_acc


def experiment_2_depth():
    """实验2：不同网络深度的影响"""
    print("\n" + "=" * 70)
    print("实验1-扩展：不同网络深度的对比分析")
    print("=" * 70)
    
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    configurations = [
        ([2, 16, 1], "浅层: [2,16,1]"),
        ([2, 16, 8, 1], "中等: [2,16,8,1]"),
        ([2, 32, 16, 8, 1], "深层: [2,32,16,8,1]"),
    ]
    
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (layer_dims, name) in enumerate(configurations):
        print(f"\n--- {name} ---")
        model = MLP(layer_dims=layer_dims, activation='relu',
                    output_activation='sigmoid', loss='bce', learning_rate=0.1)
        model.fit(X_train, y_train, X_val, y_val, epochs=2000, batch_size=64, verbose=False)
        model.restore_best()
        val_acc = model.compute_accuracy(X_val, y_val)
        results[name] = val_acc
        print(f"验证准确率: {val_acc:.4f}")
        
        # 绘制决策边界
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolor='k', cmap=plt.cm.RdYlBu, s=30)
        axes[idx].set_title(f"{name}\nVal Acc: {val_acc:.4f}", fontsize=11)
        axes[idx].set_xlabel("Feature 1")
        axes[idx].set_ylabel("Feature 2")
    
    plt.tight_layout()
    plt.savefig("./exp1_mlp/results/depth_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def experiment_3_activation():
    """实验3：不同激活函数的影响"""
    print("\n" + "=" * 70)
    print("实验1-扩展：不同激活函数的对比分析")
    print("=" * 70)
    
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    y = y.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    activations = ['relu', 'sigmoid', 'tanh']
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, act in enumerate(activations):
        print(f"\n--- 激活函数: {act} ---")
        model = MLP(layer_dims=[2, 16, 8, 1], activation=act,
                    output_activation='sigmoid', loss='bce', learning_rate=0.1)
        model.fit(X_train, y_train, X_val, y_val, epochs=2000, batch_size=64, verbose=False)
        model.restore_best()
        val_acc = model.compute_accuracy(X_val, y_val)
        results[act] = val_acc
        print(f"验证准确率: {val_acc:.4f}")
        
        # 绘制决策边界
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolor='k', cmap=plt.cm.RdYlBu, s=30)
        axes[idx].set_title(f"Activation: {act}\nVal Acc: {val_acc:.4f}", fontsize=11)
        axes[idx].set_xlabel("Feature 1")
        axes[idx].set_ylabel("Feature 2")
    
    plt.tight_layout()
    plt.savefig("./exp1_mlp/results/activation_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# =============================================
# 主程序
# =============================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  深度学习课程实验一：MLP（仅使用numpy）")
    print("#" * 70)
    
    # 实验1：基本二分类
    model, acc = experiment_1_basic()
    
    # 实验2：不同网络深度对比
    depth_results = experiment_2_depth()
    
    # 实验3：不同激活函数对比
    activation_results = experiment_3_activation()
    
    print("\n" + "=" * 70)
    print("实验一全部完成！")
    print("=" * 70)
    print("\n结果汇总：")
    print(f"  基本MLP (make_moons): 验证准确率 = {acc:.4f}")
    for name, acc_val in depth_results.items():
        print(f"  {name}: 验证准确率 = {acc_val:.4f}")
    for act, acc_val in activation_results.items():
        print(f"  激活函数 {act} (make_circles): 验证准确率 = {acc_val:.4f}")
