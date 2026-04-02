#!/usr/bin/env python
"""个性化组件示例 - 配置驱动的自定义组件训练。
 
展示:
1. 注册自定义 PyTorch 模型到注册表
2. 通过配置使用自定义组件
3. 一行调用 Trainer(config).train() 完成训练
"""
 
import sys
from pathlib import Path
 
# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
import torch
import torch.nn as nn
 
from lib.models import MODELS
from lib.loss_func import LOSSES, BaseLoss
from lib.evaluator import EVALUATORS, BaseEvaluator
from training.trainer import Trainer
 
# 数据文件路径
DATA_DIR = Path(__file__).parent.parent / 'data'
SAMPLE_DATA_PATH = DATA_DIR / 'sample_data.csv'
SMALL_DATA_PATH = DATA_DIR / 'small_data.csv'
 
 
# ============================================================================
# 1. 自定义 PyTorch 模型
# ============================================================================
 
@MODELS.register('SimpleMLP')
class SimpleMLP(nn.Module):
    """简单但完整的 MLP 模型。
 
    直接继承 nn.Module 以确保所有 PyTorch 功能正常工作。
    """
 
    def __init__(self, config=None, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
 
        # 定义真正的 PyTorch 层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self._built = True
 
    def forward(self, x):
        """前向传播。"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
 
    def build(self, input_shape=None):
        """构建模型。"""
        return self
 
 
# ============================================================================
# 2. 自定义损失函数
# ============================================================================
 
@LOSSES.register('LabelSmoothingLoss')
class LabelSmoothingLoss(BaseLoss):
    """带标签平滑的交叉熵损失。"""
 
    def __init__(self, smoothing=0.1, config=None):
        super().__init__(config)
        self.smoothing = smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)
 
    def compute(self, predictions, targets):
        """计算损失。"""
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.long)
        return self.ce_loss(predictions, targets)
 
    def __call__(self, predictions, targets):
        """使损失函数可调用。"""
        return self.compute(predictions, targets)
 
 
# ============================================================================
# 3. 自定义评估器
# ============================================================================
 
@EVALUATORS.register('Top3AccuracyEvaluator')
class Top3AccuracyEvaluator(BaseEvaluator):
    """Top-3 准确率评估器。"""
 
    def __init__(self, config=None, top_k=3):
        super().__init__(config)
        self.top_k = top_k
 
    def evaluate(self, predictions, targets):
        """评估预测结果。"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
 
        # Top-k 准确率
        top_k_preds = predictions.argsort(axis=1)[:, -self.top_k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
 
        accuracy = correct / len(targets)
        return {f'top{self.top_k}_accuracy': accuracy}
 
    def compute_metrics(self):
        """计算聚合指标。"""
        if not self.results:
            return {f'top{self.top_k}_accuracy': 0.0}
 
        key = f'top{self.top_k}_accuracy'
        values = [r[key] for r in self.results if key in r]
        return {key: sum(values) / len(values) if values else 0.0}
 
 
# ============================================================================
# 演示函数
# ============================================================================
 
def demo_custom_model():
    """演示使用自定义模型进行配置驱动训练。"""
    print("=" * 60)
    print("1. 自定义模型训练 - 配置驱动")
    print("=" * 60)
 
    num_features = 784
    num_classes = 10
 
    print(f"\n   数据文件: {SAMPLE_DATA_PATH}")
 
    # 配置使用自定义模型
    config = {
        'data': {
            'fetcher': {
                'type': 'CSVDataFetcher',
                'source': str(SAMPLE_DATA_PATH),
                'target_column': 'label'
            }
        },
        'model': {
            'type': 'SimpleMLP',  # 使用自定义注册的模型
            'input_dim': num_features,
            'hidden_dim': 256,
            'output_dim': num_classes
        },
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'Adam', 'lr': 0.001},
        'evaluator': {'type': 'AccuracyEvaluator'},
        'training': {'epochs': 5, 'batch_size': 32},
        'dataset': {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},
        'logging': {'level': 'WARNING', 'console_output': False}
    }
 
    print(f"   使用自定义模型: SimpleMLP")
 
    # 一行训练
    trainer = Trainer(config)
    history = trainer.train()
 
    # 打印结果
    print(f"\n   训练结果:")
    for i, (train, val) in enumerate(zip(history['train'], history['val'])):
        train_loss = train.get('loss', 0)
        val_acc = val.get('accuracy_top1', 0) * 100
        print(f"   Epoch {i+1}: loss={train_loss:.4f}, val_acc={val_acc:.1f}%")
 
 
def demo_custom_loss():
    """演示使用自定义损失函数进行配置驱动训练。"""
    print("\n" + "=" * 60)
    print("2. 自定义损失函数训练 (Label Smoothing) - 配置驱动")
    print("=" * 60)
 
    num_features = 100
    num_classes = 5
 
    print(f"\n   数据文件: {SMALL_DATA_PATH}")
 
    # 配置使用自定义损失
    config = {
        'data': {
            'fetcher': {
                'type': 'CSVDataFetcher',
                'source': str(SMALL_DATA_PATH),
                'target_column': 'label'
            }
        },
        'model': {
            'type': 'MLP',
            'input_dim': num_features,
            'hidden_dims': [64, 32],
            'output_dim': num_classes
        },
        'loss': {
            'type': 'LabelSmoothingLoss',  # 使用自定义损失
            'smoothing': 0.1
        },
        'optimizer': {'type': 'Adam', 'lr': 0.005},
        'evaluator': {'type': 'AccuracyEvaluator'},
        'training': {'epochs': 5, 'batch_size': 32},
        'dataset': {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},
        'logging': {'level': 'WARNING', 'console_output': False}
    }
 
    print(f"   使用自定义损失: LabelSmoothingLoss (smoothing=0.1)")
 
    # 一行训练
    trainer = Trainer(config)
    history = trainer.train()
 
    # 打印结果
    print(f"\n   训练结果:")
    for i, train in enumerate(history['train']):
        print(f"   Epoch {i+1}: loss={train.get('loss', 0):.4f}")
 
 
def demo_custom_evaluator():
    """演示使用自定义评估器进行配置驱动训练。"""
    print("\n" + "=" * 60)
    print("3. 自定义评估器训练 (Top-3 Accuracy) - 配置驱动")
    print("=" * 60)
 
    num_features = 100
    num_classes = 5
 
    print(f"\n   数据文件: {SMALL_DATA_PATH}")
 
    # 配置使用自定义评估器
    config = {
        'data': {
            'fetcher': {
                'type': 'CSVDataFetcher',
                'source': str(SMALL_DATA_PATH),
                'target_column': 'label'
            }
        },
        'model': {
            'type': 'MLP',
            'input_dim': num_features,
            'hidden_dims': [32],
            'output_dim': num_classes
        },
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'Adam', 'lr': 0.01},
        'evaluator': {
            'type': 'Top3AccuracyEvaluator',  # 使用自定义评估器
            'top_k': 3
        },
        'training': {'epochs': 5, 'batch_size': 32},
        'dataset': {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},
        'logging': {'level': 'WARNING', 'console_output': False}
    }
 
    print(f"   使用自定义评估器: Top3AccuracyEvaluator")
 
    # 一行训练
    trainer = Trainer(config)
    history = trainer.train()
 
    # 打印结果
    print(f"\n   训练结果:")
    for i, (train, val) in enumerate(zip(history['train'], history['val'])):
        train_loss = train.get('loss', 0)
        top3_acc = val.get('top3_accuracy', 0) * 100
        print(f"   Epoch {i+1}: loss={train_loss:.4f}, top3_acc={top3_acc:.1f}%")
 
 
def demo_list_registered_components():
    """列出所有已注册的组件。"""
    print("\n" + "=" * 60)
    print("4. 已注册组件列表")
    print("=" * 60)
 
    registries = {
        'MODELS': MODELS,
        'LOSSES': LOSSES,
        'EVALUATORS': EVALUATORS,
    }
 
    for name, registry in registries.items():
        registered = registry.list_registered()
        print(f"\n{name}: {', '.join(registered)}")
 
 
def main():
    """运行所有示例。"""
    print("\n" + "=" * 60)
    print("个性化组件示例 - 配置驱动的自定义组件训练")
    print("=" * 60 + "\n")
 
    demo_custom_model()
    demo_custom_loss()
    demo_custom_evaluator()
    demo_list_registered_components()
 
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)
 
 
if __name__ == '__main__':
    main()