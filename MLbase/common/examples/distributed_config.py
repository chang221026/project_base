#!/usr/bin/env python
"""配置驱动的分布式训练示例。
 
展示:
1. distributed 配置项
2. 多种分布式模式 (auto, single, single_multi, multi_multi)
3. 策略配置
4. torchrun 启动命令示例
"""
 
import os
import sys
from pathlib import Path
 
# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
from utils.config_management import Config
from training.trainer import Trainer
 
 
def demo_distributed_modes():
    """演示分布式模式配置。"""
    print("=" * 60)
    print("1. 分布式模式配置")
    print("=" * 60)
 
    modes = {
        'auto': {
            'description': '自动检测并选择最佳模式',
            'use_case': '推荐使用，自动适配环境'
        },
        'single': {
            'description': '单机单卡训练',
            'use_case': '开发调试、小规模实验'
        },
        'single_multi': {
            'description': '单机多卡训练',
            'use_case': '单台服务器多GPU/NPU'
        },
        'multi_multi': {
            'description': '多机多卡训练',
            'use_case': '大规模分布式训练'
        }
    }
 
    print("\n可用模式:")
    for mode, info in modes.items():
        print(f"\n  {mode}:")
        print(f"    描述: {info['description']}")
        print(f"    场景: {info['use_case']}")
 
    print("\n配置示例:")
    print("""
  distributed:
    mode: auto  # auto, single, single_multi, multi_multi
    backend: nccl
    init_method: env://
    """)
 
 
def demo_single_gpu_config():
    """演示单机单卡配置。"""
    print("\n" + "=" * 60)
    print("2. 单机单卡配置")
    print("=" * 60)
 
    config = Config.from_dict({
        'model': {'type': 'MLP'},
        'optimizer': {'type': 'Adam'},
        'training': {'epochs': 10},
        'distributed': {
            'mode': 'single'
        }
    })
 
    print("配置:")
    print(f"  mode = {config.get('distributed.mode')}")
 
    print("\n启动命令:")
    print("  python train.py --config config.yaml")
 
 
def demo_single_multi_gpu_config():
    """演示单机多卡配置。"""
    print("\n" + "=" * 60)
    print("3. 单机多卡配置")
    print("=" * 60)
 
    config = Config.from_dict({
        'model': {'type': 'MLP'},
        'optimizer': {'type': 'Adam'},
        'training': {'epochs': 10},
        'distributed': {
            'mode': 'single_multi',
            'backend': 'nccl',
            'strategies': [
                {'type': 'DDP'}
            ]
        }
    })
 
    print("配置:")
    print(f"  mode = {config.get('distributed.mode')}")
    print(f"  backend = {config.get('distributed.backend')}")
 
    print("\n启动命令 (torchrun):")
    print("  torchrun --nproc_per_node=4 train.py --config config.yaml")
    print("\n或使用 python 启动:")
    print("  python -m torch.distributed.launch \\")
    print("    --nproc_per_node=4 \\")
    print("    train.py --config config.yaml")
 
 
def demo_multi_machine_config():
    """演示多机多卡配置。"""
    print("\n" + "=" * 60)
    print("4. 多机多卡配置")
    print("=" * 60)
 
    config = Config.from_dict({
        'model': {'type': 'MLP'},
        'optimizer': {'type': 'Adam'},
        'training': {'epochs': 10},
        'distributed': {
            'mode': 'multi_multi',
            'backend': 'nccl',
            'init_method': 'env://',
            'strategies': [
                {'type': 'DDP'}
            ]
        }
    })
 
    print("配置:")
    print(f"  mode = {config.get('distributed.mode')}")
    print(f"  backend = {config.get('distributed.backend')}")
    print(f"  init_method = {config.get('distributed.init_method')}")
 
    print("\n环境变量设置:")
    print("  export MASTER_ADDR=10.0.0.1      # 主节点地址")
    print("  export MASTER_PORT=29500         # 通信端口")
    print("  export WORLD_SIZE=8              # 总进程数 (节点数 * 每节点GPU数)")
    print("  export RANK=0                    # 当前节点全局排名 (每个节点不同)")
    print("  export LOCAL_RANK=0              # 本地进程排名")
 
    print("\n启动命令 (在每个节点上执行):")
    print("\n  # 节点 0 (主节点):")
    print("  RANK=0 torchrun --nproc_per_node=4 --nnodes=2 \\")
    print("    --node_rank=0 --master_addr=10.0.0.1 \\")
    print("    train.py --config config.yaml")
 
    print("\n  # 节点 1:")
    print("  RANK=1 torchrun --nproc_per_node=4 --nnodes=2 \\")
    print("    --node_rank=1 --master_addr=10.0.0.1 \\")
    print("    train.py --config config.yaml")
 
 
def demo_strategy_config():
    """演示策略配置。"""
    print("\n" + "=" * 60)
    print("5. 策略配置")
    print("=" * 60)
 
    config = Config.from_dict({
        'distributed': {
            'mode': 'auto',
            'strategies': [
                {'type': 'DDP'},
                {'type': 'FP16'}
            ]
        }
    })
 
    print("策略配置:")
    print(f"  strategies = {[s['type'] for s in config.get('distributed.strategies')]}")
 
    print("\n可用策略:")
    strategies = {
        'DDP': '分布式数据并行 (DistributedDataParallel)',
        'FP16': '混合精度训练 (需要设备支持)',
        'GradientAccumulation': '梯度累积 (增大有效 batch size)',
        'TorchCompile': 'PyTorch 2.0 编译优化'
    }
 
    for name, desc in strategies.items():
        print(f"  {name}: {desc}")
 
    print("\n示例配置:")
    print("""
  distributed:
    mode: single_multi
    strategies:
      - type: DDP
      - type: FP16
        loss_scale: dynamic
      - type: GradientAccumulation
        accumulation_steps: 4
    """)
 
 
def demo_auto_mode():
    """演示自动模式。"""
    print("\n" + "=" * 60)
    print("6. 自动模式 (推荐)")
    print("=" * 60)
 
    config = Config.from_dict({
        'distributed': {
            'mode': 'auto'
        }
    })
 
    print("配置:")
    print(f"  mode = {config.get('distributed.mode')}")
 
    print("\n自动模式行为:")
    print("  1. 检测可用设备 (NPU -> GPU -> CPU)")
    print("  2. 根据设备数量选择模式:")
    print("     - 1 个设备: single 模式")
    print("     - 多设备同机: single_multi 模式")
    print("     - 多机多设备: multi_multi 模式")
    print("  3. 自动创建合适的策略链")
 
    print("\n推荐使用 auto 模式，框架自动适配环境")
 
 
def demo_device_preference():
    """演示设备偏好配置。"""
    print("\n" + "=" * 60)
    print("7. 设备偏好配置")
    print("=" * 60)
 
    config = Config.from_dict({
        'device': {
            'preferred_device': 'npu'  # 或 'gpu', 'cpu', null (自动)
        },
        'distributed': {
            'mode': 'auto'
        }
    })
 
    print("配置:")
    print(f"  preferred_device = {config.get('device.preferred_device')}")
 
    print("\n设备优先级 (自动模式):")
    print("  1. NPU (华为昇腾)")
    print("  2. GPU (NVIDIA CUDA)")
    print("  3. CPU (回退)")
 
    print("\n强制使用特定设备:")
    print("  preferred_device: 'gpu'  # 强制使用 GPU")
 
 
def main():
    """运行所有示例。"""
    print("\n" + "=" * 60)
    print("配置驱动的分布式训练示例")
    print("=" * 60 + "\n")
 
    demo_distributed_modes()
    demo_single_gpu_config()
    demo_single_multi_gpu_config()
    demo_multi_machine_config()
    demo_strategy_config()
    demo_auto_mode()
    demo_device_preference()
 
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)
 
    print("\n快速启动指南:")
    print("\n  # 单机训练")
    print("  python train.py --config config.yaml")
    print("\n  # 单机多卡")
    print("  torchrun --nproc_per_node=4 train.py --config config.yaml")
    print("\n  # 多机多卡")
    print("  torchrun --nnodes=2 --nproc_per_node=4 \\")
    print("    --master_addr=10.0.0.1 train.py --config config.yaml")
 
 
if __name__ == '__main__':
    main()