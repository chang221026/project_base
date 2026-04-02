#!/usr/bin/env python
"""配置管理示例。
 
展示 Config 类的完整功能:
1. 从 YAML/JSON 文件加载配置
2. 从字典创建配置
3. 环境变量覆盖
4. 默认值设置
5. 配置合并
6. 嵌套值访问 (点分隔符)
7. instantiate() 动态实例化
"""
 
import os
import sys
from pathlib import Path
 
# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
from utils.config_management import Config, load_config, instantiate
 
 
def demo_file_loading():
    """演示从文件加载配置。"""
    print("=" * 60)
    print("1. 从文件加载配置")
    print("=" * 60)
 
    # 从 YAML 文件加载
    config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    if config_path.exists():
        config = Config.from_file(config_path)
        print(f"从 YAML 加载成功: {config_path}")
        print(f"training.epochs = {config.get('training.epochs')}")
        print(f"model.type = {config.get('model.type')}")
    else:
        print(f"配置文件不存在: {config_path}")
 
 
def demo_dict_creation():
    """演示从字典创建配置。"""
    print("\n" + "=" * 60)
    print("2. 从字典创建配置")
    print("=" * 60)
 
    config_dict = {
        'model': {
            'type': 'MLP',
            'input_dim': 784,
            'hidden_dims': [128, 64],
            'output_dim': 10
        },
        'training': {
            'epochs': 100,
            'batch_size': 32
        }
    }
 
    config = Config.from_dict(config_dict)
    print(f"创建配置: {config}")
    print(f"model.type = {config.get('model.type')}")
    print(f"training.batch_size = {config.get('training.batch_size')}")
 
 
def demo_env_override():
    """演示环境变量覆盖。"""
    print("\n" + "=" * 60)
    print("3. 环境变量覆盖")
    print("=" * 60)
 
    # 设置环境变量 (前缀 ML_)
    os.environ['ML_TRAINING_EPOCHS'] = '50'
    os.environ['ML_TRAINING_BATCH_SIZE'] = '64'
    os.environ['ML_MODEL_HIDDEN_DIMS'] = '[256, 128]'
 
    config_dict = {
        'training': {'epochs': 10, 'batch_size': 16},
        'model': {'type': 'MLP'}
    }
 
    config = Config.from_dict(config_dict)
    config.apply_env_overrides('ML_')
 
    print(f"原始 training.epochs = 10, 覆盖后 = {config.get('training.epochs')}")
    print(f"原始 training.batch_size = 16, 覆盖后 = {config.get('training.batch_size')}")
    print(f"model.hidden_dims = {config.get('model.hidden_dims')}")
 
    # 清理环境变量
    del os.environ['ML_TRAINING_EPOCHS']
    del os.environ['ML_TRAINING_BATCH_SIZE']
    del os.environ['ML_MODEL_HIDDEN_DIMS']
 
 
def demo_default_values():
    """演示默认值设置。"""
    print("\n" + "=" * 60)
    print("4. 默认值设置")
    print("=" * 60)
 
    config_dict = {
        'training': {'epochs': 50}
    }
 
    default_dict = {
        'training': {'epochs': 10, 'batch_size': 32},
        'model': {'type': 'MLP', 'input_dim': 784}
    }
 
    config = Config.from_dict(config_dict)
    config.set_default(default_dict)
 
    print("配置值 (优先于默认值):")
    print(f"  training.epochs = {config.get('training.epochs')} (配置值 50)")
    print("默认值 (配置中不存在时使用):")
    print(f"  training.batch_size = {config.get('training.batch_size')} (默认值 32)")
    print(f"  model.type = {config.get('model.type')} (默认值 MLP)")
 
 
def demo_config_merge():
    """演示配置合并。"""
    print("\n" + "=" * 60)
    print("5. 配置合并")
    print("=" * 60)
 
    config1 = Config.from_dict({
        'model': {'type': 'MLP', 'hidden_dims': [128]},
        'training': {'epochs': 10}
    })
 
    config2 = {
        'model': {'hidden_dims': [256, 128]},  # 覆盖
        'training': {'batch_size': 64},         # 新增
        'optimizer': {'type': 'Adam'}           # 新增
    }
 
    config1.merge(config2)
 
    print("合并后配置:")
    print(f"  model.type = {config1.get('model.type')} (保留)")
    print(f"  model.hidden_dims = {config1.get('model.hidden_dims')} (覆盖)")
    print(f"  training.epochs = {config1.get('training.epochs')} (保留)")
    print(f"  training.batch_size = {config1.get('training.batch_size')} (新增)")
    print(f"  optimizer.type = {config1.get('optimizer.type')} (新增)")
 
 
def demo_nested_access():
    """演示嵌套值访问。"""
    print("\n" + "=" * 60)
    print("6. 嵌套值访问 (点分隔符)")
    print("=" * 60)
 
    config = Config.from_dict({
        'data': {
            'fetcher': {
                'type': 'CSVDataFetcher',
                'source': './data.csv'
            },
            'processors': [
                {'type': 'StandardScaler'},
                {'type': 'Normalizer'}
            ]
        },
        'dataset': {
            'train_ratio': 0.7,
            'val_ratio': 0.15
        }
    })
 
    print("访问嵌套值:")
    print(f"  data.fetcher.type = {config.get('data.fetcher.type')}")
    print(f"  data.fetcher.source = {config.get('data.fetcher.source')}")
    print(f"  dataset.train_ratio = {config.get('dataset.train_ratio')}")
 
    # 使用方括号语法
    print("\n使用方括号语法:")
    print(f"  config['data.processors'] = {config['data.processors']}")
 
    # 设置值
    config.set('data.fetcher.source', './new_data.csv')
    print(f"\n修改后 data.fetcher.source = {config.get('data.fetcher.source')}")
 
 
def demo_instantiate():
    """演示 instantiate() 动态实例化。"""
    print("\n" + "=" * 60)
    print("7. instantiate() 动态实例化")
    print("=" * 60)
 
    # 使用 _target_ 动态实例化任何类
    # 这里用标准库的类作为示例
    from collections import defaultdict
 
    config = {
        '_target_': 'collections.defaultdict',
    }
 
    obj = instantiate(config)
    # 设置 default_factory
    obj.default_factory = list
    print(f"实例化 defaultdict: {type(obj).__name__}")
    obj['key'].append('value')
    print(f"  obj['key'] = {obj['key']}")
 
    # 嵌套实例化
    nested_config = {
        '_target_': 'collections.OrderedDict',
        # 其他参数...
    }
 
    print("\n支持嵌套配置的实例化")
 
 
def demo_load_config_helper():
    """演示 load_config 便捷函数。"""
    print("\n" + "=" * 60)
    print("8. load_config() 便捷函数")
    print("=" * 60)
 
    default_config = {
        'training': {'epochs': 10, 'batch_size': 32},
        'model': {'type': 'MLP'}
    }
 
    # 设置环境变量
    os.environ['ML_TRAINING_EPOCHS'] = '200'
 
    config = load_config(
        filepath=None,           # 不从文件加载
        default_config=default_config,
        env_prefix='ML_'
    )
 
    print(f"training.epochs = {config.get('training.epochs')} (环境变量覆盖)")
    print(f"training.batch_size = {config.get('training.batch_size')} (默认值)")
    print(f"model.type = {config.get('model.type')} (默认值)")
 
    del os.environ['ML_TRAINING_EPOCHS']
 
 
def main():
    """运行所有示例。"""
    print("\n" + "=" * 60)
    print("配置管理 (Config Management) 示例")
    print("=" * 60 + "\n")
 
    demo_file_loading()
    demo_dict_creation()
    demo_env_override()
    demo_default_values()
    demo_config_merge()
    demo_nested_access()
    demo_instantiate()
    demo_load_config_helper()
 
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)
 
 
if __name__ == '__main__':
    main()