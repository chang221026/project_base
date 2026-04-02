#!/usr/bin/env python
"""配置驱动的默认训练示例 - 最简洁用法。
 
展示:
1. 使用静态 CSV 数据文件
2. 配置 data.fetcher 指向数据文件
3. 一行调用 Trainer(config).train() 完成所有操作
"""
 
import sys
from pathlib import Path
 
# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
from training.trainer import Trainer
 
# 数据文件路径
DATA_DIR = Path(__file__).parent.parent / 'data'
SAMPLE_DATA_PATH = DATA_DIR / 'sample_data.csv'
 
 
def main():
    """运行配置驱动的训练示例。"""
    print("=" * 60)
    print("配置驱动的默认训练示例")
    print("=" * 60)
 
    # =========================================================================
    # 1. 使用静态数据文件
    # =========================================================================
    print("\n1. 加载数据文件...")
 
    num_features = 784
    num_classes = 10
 
    print(f"   数据文件: {SAMPLE_DATA_PATH}")
    print(f"   特征维度: {num_features}, 类别数: {num_classes}")
 
    # =========================================================================
    # 2. 创建完整配置
    # =========================================================================
    print("\n2. 创建配置...")
 
    config = {
        'data': {
            'fetcher': {
                'type': 'CSVDataFetcher',
                'source': str(SAMPLE_DATA_PATH),
                'target_column': 'label'
            }
        },
        'model': {
            'type': 'MLP',
            'input_dim': num_features,
            'hidden_dims': [256, 128],
            'output_dim': num_classes,
            'activation': 'relu'
        },
        'loss': {
            'type': 'CrossEntropyLoss'
        },
        'optimizer': {
            'type': 'Adam',
            'lr': 0.001,
            'weight_decay': 0.0001
        },
        'evaluator': {
            'type': 'AccuracyEvaluator',
            'top_k': 1
        },
        'training': {
            'epochs': 5,
            'batch_size': 32
        },
        'dataset': {
            'train_ratio': 0.8,
            'val_ratio': 0.2,
            'test_ratio': 0.0,
            'shuffle': True,
            'random_seed': 42
        },
        'distributed': {
            'mode': 'auto'  # AUTO mode: auto-detect devices (NPU > GPU > CPU) and select optimal training mode
        },
        'logging': {
            'level': 'WARNING',
            'console_output': False
        }
    }
 
    print("   模型: MLP (784 -> 256 -> 128 -> 10)")
    print("   损失函数: CrossEntropyLoss")
    print("   优化器: Adam (lr=0.001)")
    print("   训练参数: epochs=5, batch_size=32")
    print("   数据划分: train=80%, val=20%")
 
    # =========================================================================
    # 3. 一行训练 - Trainer 自动完成所有操作
    # =========================================================================
    print("\n3. 开始训练 (Trainer 自动获取数据、预处理、划分、训练)...")
 
    trainer = Trainer(config)
    history = trainer.train()
 
    # =========================================================================
    # 4. 打印训练结果
    # =========================================================================
    print("\n4. 训练结果:")
    print("=" * 60)
 
    print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Val Accuracy':<15}")
    print("-" * 40)
 
    for i, (train, val) in enumerate(zip(history['train'], history['val'])):
        train_loss = train.get('loss', 0)
        val_acc = val.get('accuracy_top1', 0) * 100
        print(f"{i+1:<8} {train_loss:<15.4f} {val_acc:<15.2f}%")
 
    # 计算总体改进
    if len(history['train']) > 1:
        first_loss = history['train'][0].get('loss', 0)
        last_loss = history['train'][-1].get('loss', 0)
        improvement = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0
        print("-" * 40)
        print(f"\n训练 loss 改进: {first_loss:.4f} -> {last_loss:.4f} ({improvement:.1f}%)")
 
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
 
 
if __name__ == '__main__':
    main()