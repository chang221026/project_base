#!/usr/bin/env python
"""检查点重训示例 - 配置驱动的检查点保存和恢复。
 
展示:
1. 通过配置自动保存检查点
2. 使用 Trainer.save() 手动保存
3. 使用 Trainer.load() 恢复训练
4. 一行调用 Trainer(config).train() 完成训练
"""
 
import sys
import tempfile
import shutil
from pathlib import Path
 
# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
from training.trainer import Trainer
 
# 数据文件路径
DATA_DIR = Path(__file__).parent.parent / 'data'
SMALL_DATA_PATH = DATA_DIR / 'small_data.csv'
 
 
# ============================================================================
# 演示函数
# ============================================================================
 
def demo_checkpoint_training():
    """演示检查点保存和恢复训练。"""
    print("=" * 60)
    print("检查点保存和恢复训练示例 - 配置驱动")
    print("=" * 60)
 
    num_features = 100
    num_classes = 5
 
    # 创建临时目录用于检查点
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_dir = temp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"\n   数据文件: {SMALL_DATA_PATH}")
    print(f"   检查点目录: {checkpoint_dir}")
 
    try:
        # =========================================
        # 第一阶段: 训练并保存检查点
        # =========================================
        print("\n" + "-" * 60)
        print("第一阶段: 训练 3 个 epoch 并保存检查点")
        print("-" * 60)
 
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
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam', 'lr': 0.005},
            'evaluator': {'type': 'AccuracyEvaluator'},
            'training': {'epochs': 3, 'batch_size': 32},
            'dataset': {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},
            'logging': {'level': 'WARNING', 'console_output': False},
            'hooks': {
                'checkpoint': {
                    'type': 'checkpoint',
                    'checkpoint_dir': str(checkpoint_dir),
                    'save_interval': 1,
                    'save_best': True,
                    'monitor': 'loss'
                }
            }
        }
 
        # 一行训练
        trainer = Trainer(config)
        history1 = trainer.train()
 
        # 打印第一阶段结果
        print(f"\n   第一阶段训练结果:")
        for i, train in enumerate(history1['train']):
            print(f"   Epoch {i+1}: loss={train.get('loss', 0):.4f}")
 
        # 手动保存最终模型
        final_checkpoint = checkpoint_dir / 'final_model.pth'
        trainer.save(str(final_checkpoint))
        print(f"\n   最终模型已保存: {final_checkpoint.name}")
 
        # 列出保存的检查点
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        print(f"\n   保存的检查点:")
        for cp in sorted(checkpoints):
            print(f"   - {cp.name}")
 
        # =========================================
        # 第二阶段: 从检查点恢复并继续训练
        # =========================================
        print("\n" + "-" * 60)
        print("第二阶段: 从检查点恢复并继续训练 2 个 epoch")
        print("-" * 60)
 
        # 创建新配置 - 增加 epochs
        config2 = config.copy()
        config2['training'] = {'epochs': 5, 'batch_size': 32}
        # 移除钩子避免重复保存
        config2['hooks'] = {}
 
        # 创建新的 Trainer 并加载检查点
        trainer2 = Trainer(config2)
        trainer2.load(str(checkpoint_dir / 'best_checkpoint.pth'))
 
        # 继续训练
        history2 = trainer2.train()
 
        # 打印第二阶段结果
        print(f"\n   第二阶段训练结果:")
        for i, train in enumerate(history2['train']):
            print(f"   Epoch {i+1}: loss={train.get('loss', 0):.4f}")
 
        # =========================================
        # 打印完整的训练历史
        # =========================================
        print("\n" + "-" * 60)
        print("完整训练历史")
        print("-" * 60)
 
        print(f"\n{'阶段':<10} {'Epoch':<8} {'Train Loss':<15}")
        print("-" * 40)
 
        for i, train in enumerate(history1['train']):
            print(f"{'阶段 1':<10} {i+1:<8} {train.get('loss', 0):<15.4f}")
 
        for i, train in enumerate(history2['train']):
            print(f"{'阶段 2':<10} {i+1:<8} {train.get('loss', 0):<15.4f}")
 
    finally:
        # 清理检查点目录
        shutil.rmtree(temp_dir, ignore_errors=True)
 
 
def demo_best_model():
    """演示保存和加载最佳模型。"""
    print("\n" + "=" * 60)
    print("最佳模型保存和加载示例 - 配置驱动")
    print("=" * 60)
 
    num_features = 100
    num_classes = 5
 
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_dir = temp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"\n   数据文件: {SMALL_DATA_PATH}")
 
    try:
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
            'evaluator': {'type': 'AccuracyEvaluator'},
            'training': {'epochs': 5, 'batch_size': 32},
            'dataset': {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},
            'logging': {'level': 'WARNING', 'console_output': False},
            'hooks': {
                'checkpoint': {
                    'type': 'checkpoint',
                    'checkpoint_dir': str(checkpoint_dir),
                    'save_interval': 1,
                    'save_best': True,
                    'monitor': 'loss',
                    'mode': 'min'
                }
            }
        }
 
        print("\n   训练并追踪最佳模型...")
 
        # 一行训练
        trainer = Trainer(config)
        history = trainer.train()
 
        # 打印训练结果
        print(f"\n   训练结果:")
        best_loss = float('inf')
        best_epoch = 0
        for i, train in enumerate(history['train']):
            loss = train.get('loss', 0)
            if loss < best_loss:
                best_loss = loss
                best_epoch = i + 1
            print(f"   Epoch {i+1}: loss={loss:.4f}")
 
        print(f"\n   最佳模型:")
        print(f"   最佳 Epoch: {best_epoch}")
        print(f"   最佳 Loss: {best_loss:.4f}")
 
        # 加载最佳模型
        print("\n   加载最佳模型验证...")
        config2 = config.copy()
        config2['hooks'] = {}
        config2['training'] = {'epochs': 1, 'batch_size': 32}
 
        trainer2 = Trainer(config2)
        trainer2.load(str(checkpoint_dir / 'best_checkpoint.pth'))
 
        # 手动评估
        print(f"   最佳模型已加载，可用于继续训练或预测")
 
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
 
 
def demo_manual_save_load():
    """演示手动保存和加载模型。"""
    print("\n" + "=" * 60)
    print("手动保存和加载示例 - 配置驱动")
    print("=" * 60)
 
    num_features = 100
    num_classes = 5
 
    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / 'model.pth'
 
    print(f"\n   数据文件: {SMALL_DATA_PATH}")
 
    try:
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
                'hidden_dims': [16],
                'output_dim': num_classes
            },
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam', 'lr': 0.01},
            'evaluator': {'type': 'AccuracyEvaluator'},
            'training': {'epochs': 3, 'batch_size': 32},
            'dataset': {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},
            'logging': {'level': 'WARNING', 'console_output': False}
        }
 
        # 训练
        print("\n   训练模型...")
        trainer = Trainer(config)
        history = trainer.train()
 
        # 手动保存
        trainer.save(str(model_path))
        print(f"   模型已保存: {model_path.name}")
 
        # 加载模型
        config2 = config.copy()
        trainer2 = Trainer(config2)
        trainer2.load(str(model_path))
        print(f"   模型已加载")
 
        # 使用加载的模型进行预测
        import numpy as np
        test_data = np.random.randn(5, num_features).astype(np.float32)
        predictions = trainer2.predict(test_data)
        if predictions is not None:
            print(f"   预测完成: shape={predictions.shape}")
 
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
 
 
def main():
    """运行所有示例。"""
    print("\n" + "=" * 60)
    print("检查点重训示例 - 配置驱动")
    print("=" * 60 + "\n")
 
    demo_checkpoint_training()
    demo_best_model()
    demo_manual_save_load()
 
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)
 
 
if __name__ == '__main__':
    main()