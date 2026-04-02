# Common 框架
 
<!-- prettier-ignore-start -->
[![PyPI version](https://img.shields.io/pypi/v/common.svg)](https://pypi.org/project/common/)
[![Python version](https://img.shields.io/pypi/python_version/common.svg)](https://pypi.org/project/common/)
[![License](https://img.shields.io/pypi/l/common.svg)](https://pypi.org/project/common/)
<!-- prettier-ignore-end -->
 
Common 是一个企业级机器学习训练框架，专为 AI 开发者设计。它提供了完整的训练流程支持，从数据处理到模型训练、从分布式训练到检查点管理，帮助开发者快速构建和部署机器学习模型。
 
---
 
## 一、框架简介
 
### 1.1 核心功能
 
- **统一入口（Trainer）**：通过 Trainer 类一站式完成所有训练流程
- **配置驱动的训练流程**：通过 YAML/JSON 配置文件定义训练流程，支持环境变量覆盖和动态实例化
- **自动设备检测**：自动检测可用硬件（NPU/GPU/CPU），无需手动配置
- **分布式训练支持**：原生支持分布式数据并行（DDP），适配多种通信后端
- **完整的组件库**：内置模型、损失函数、优化器、评估器等常用组件
- **可扩展的注册机制**：通过注册表系统轻松扩展新组件
 
### 1.2 设计目标
 
- **简化 ML 训练流程**：减少样板代码，聚焦核心算法开发
- **高内聚低耦合**：通过Facade模式统一入口，内部模块解耦
- **提高代码复用性**：通过配置和注册机制实现组件复用
- **支持大规模分布式训练**：一键启动分布式训练
- **零成本硬件迁移**：自动检测硬件，无需修改代码
 
### 1.3 目标人群
 
- **AI 研究人员**：快速验证算法想法
- **ML 工程师**：生产环境模型训练与部署
- **企业级用户**：大规模分布式训练场景
- **学生/入门者**：学习机器学习的完整流程
 
### 1.4 适用场景
 
- **学术实验**：快速验证新算法，频繁调整参数
- **生产部署**：标准化训练流程，可复现结果
- **大规模分布式训练**：多卡/多机训练场景
- **模型微调**：基于预训练模型进行微调
- **强化学习**：自定义强化学习算法训练
 
---
 
## 二、框架架构图
 
```
部署层与监控层未实现
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户层                                    │
│                    (Trainer 统一入口)                                  │
│                    (编写配置 / Python API )                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              配置层                                     │
│         ┌──────────────┬──────────────┬──────────────┐                │
│         │   YAML/JSON   │   环境变量   │   默认值    │                │
│         └──────────────┴──────────────┴──────────────┘                │
│                         Config 管理器                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              (Trainer 内部)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   DataFacade│  │TrainingFacade│  │             │  │    Hook      │ │
│  │             │  │ (含Distributed)│  │             │  │    Layer    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              库层                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │  Models  │ │  Losses  │ │Optimizers│ │Evaluators│ │ Hooks (...)│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
│                     (通过 Registry 扩展)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              工具层                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │
│  │DeviceManager │ │  Distributed │ │   Registry  │                  │
│  │  (NPU/GPU)   │ │   Manager    │ │  (Component)│                  │
│  └──────────────┘ └──────────────┘ └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              硬件层                                     │
│                    [NPU] ── [GPU] ── [CPU]                              │
└─────────────────────────────────────────────────────────────────────────┘
```
 
### 用户使用流程
 
```
用户代码
    │
    ▼
Trainer(config)  ←── 统一入口
    │
    ├─→ Config 加载配置
    ├─→ DataFacade 设置数据管道
    ├─→ TrainingFacade 构建算法（含分布式引擎隐藏内部）
    ├─→ Hooks 注册钩子
    │
    ▼
trainer.train()  ←── 一行命令启动训练
```
 
---
 
## 三、重要功能介绍
 
### 3.1 Trainer 统一入口
 
Trainer 是框架的 Facade（门面）类，提供统一的训练入口。
 
位于 `src/training/trainer.py`
 
```python
from src.training import Trainer, train
 
# 方法1: 从配置文件（推荐）
trainer = Trainer("config.yaml")
history = trainer.train()
 
# 方法2: 从字典配置
config = {
    'model': {'type': 'MLP', 'input_dim': 784, 'output_dim': 10},
    'data': {'fetcher': {'type': 'CSVDataFetcher', 'source': 'data.csv'}},
    'training': {'epochs': 10}
}
trainer = Trainer(config)
history = trainer.train()
 
# 方法3: 一行代码（最简）
history = train("config.yaml")
```
 
#### Trainer 功能
 
- 自动配置加载和验证
- 自动数据预处理管道设置（fetcher, analyzer, processors等）
- 自动训练/验证/测试集划分
- 自定义组件注册
- 算法自动构建
- 分布式训练设置
- 钩子管理
- 训练和评估
 
### 3.2 Facade 门面架构
 
Common 框架采用 **Facade 模式**构建，从顶层入口到内部组件层层递进：
 
```
┌─────────────────────────────────────────────────────────────────────┐
│                         Trainer (顶层入口)                          │
│                  配置加载 → 自动分布式启动 → 流程编排                │
└─────────────────────────────┬───────────────────────────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
┌─────────────────────────┐   ┌─────────────────────────────────────────┐
│    DataFacade             │   │    TrainingFacade                    │
│    (数据门面)             │   │    (训练门面)                        │
│                          │   │                                      │
│ 数据获取/处理/分割        │   │ 模型/损失/优化器/评估器/算法           │
└────────────┬─────────────┘   └────────────┬────────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      扩展层 (Registry 注册表)                       │
│  DATA_FETCHERS | DATA_ANALYZERS | DATA_PROCESSORS | MODELS | ...   │
└─────────────────────────────────────────────────────────────────────┘
```
 
#### DataFacade 数据门面
 
位于 `src/training/data_facade.py`
 
```python
from training import DataFacade
 
# 创建数据门面
facade = DataFacade(config)
facade.setup()
 
# 获取数据加载器
train_loader, val_loader, test_loader = facade.get_data_loaders()
```
 
#### DataFacade 内部架构
 
```
DataFacade (入口)
    │
    ├─> DataPreprocessingPipeline (数据处理层)
    │       ├── fetcher: DATA_FETCHERS (lib/data_fetching)
    │       ├── analyzer: DATA_ANALYZERS (lib/data_analysis)
    │       ├── processors: DATA_PROCESSORS (lib/data_processing)
    │       ├── constructors: FEATURE_CONSTRUCTORS
    │       ├── selectors: FEATURE_SELECTORS
    │       └── imbalance_handler: IMBALANCE_HANDLERS
    │
    └─> DatasetBuilder (数据集构建层)
            ├── 装载数据
            └── 划分 train/val/test → 返回 Dataset
    │
    └─> DataLoader (批量迭代层)
            └── 包装 Dataset，提供批量迭代
```
 
#### TrainingFacade 训练门面
 
位于 `src/training/training_facade.py`
 
```python
from training import TrainingFacade
 
# 创建训练门面
facade = TrainingFacade(config)
facade.setup()
 
# 执行训练
history = facade.train(train_loader, val_loader)
```
 
#### TrainingFacade 内部架构
 
```
TrainingFacade (入口)
    │
    ├─> DistributedEngine (分布式引擎)
    │       ├── initialize() [自动检测设备]
    │       ├── prepare_dataloader() [添加 DistributedSampler]
    │       └── prepare_model() [DDP 包装模型]
    │
    └─> Algorithm (算法层)
            ├── model: MODELS (lib/models)
            ├── loss_fn: LOSSES (lib/loss_func)
            ├── optimizer: OPTIMIZERS (lib/optimizer)
            ├── evaluator: EVALUATORS (lib/evaluator)
            └── hooks: HOOKS (training/hook)
```
 
---
 
### 3.3 数据处理完整流程
 
```
用户代码
    │
    ▼
Trainer(config)  ←── 统一入口
    │
    ├─> DataFacade.setup()
    │       │
    │       ├─> DataPreprocessingPipeline.setup()
    │       │       └── 配置 fetcher/analyzer/processors 等
    │       │
    │       └─> DatasetBuilder.setup()
    │               └── 配置 train/val/test 划分
    │
    ├─> TrainingFacade.setup()
    │       │
    │       ├─> _setup_distributed() (内部隐藏)
    │       │       └── DistributedEngine 初始化
    │       │
    │       └─> Algorithm.setup()
    │               └── 配置 model/loss/optimizer/evaluator/hooks
    │
    ▼
trainer.train()  ←── 启动训练
    │
    ├─> DataFacade.get_data_loaders()
    │       │
    │       ├─> fetcher.fetch() → 获取原始数据
    │       │
    │       ├─> pipeline.run() → 数据预处理
    │       │
    │       ├─> builder.build() → 数据划分
    │       │
    │       └─> DataLoader() → 批量迭代
    │
    ├─> TrainingFacade.train()
    │       │
    │       ├─> engine.prepare_dataloader() → DDP 分布式采样
    │       │
    │       ├─> engine.prepare_model() → DDP 包装模型
    │       │
    │       └─> algorithm.fit() → 模型训练
    │
    └─> 返回训练历史
```
 
### 3.4 配置驱动
 
配置驱动是 Common 框架的核心特性，通过声明式配置定义完整的训练流程。
 
#### 配置加载优先级（从高到低）
 
1. **环境变量** - 以指定前缀开头（如 `ML_`）
2. **配置文件** - YAML/JSON 文件
3. **默认值** - 代码中定义的默认值
 
#### 核心类：Config
 
位于 `src/utils/config_management.py`
 
```python
from src.utils import Config, load_config
 
# 从文件加载配置
config = load_config('config.yaml')
config = Config.from_file('config.json')
 
# 从字典创建
config = Config.from_dict({
    'training': {'epochs': 10}
})
 
# 环境变量覆盖（Trainer 自动调用）
config.apply_env_overrides('ML_')
 
# 嵌套值访问
epochs = config.get('training.epochs')
hidden_dims = config.get('model.hidden_dims')
 
# 动态实例化
from src.utils import instantiate
model = instantiate(config.get('model'))
```
 
#### 动态实例化
 
使用 `_target_` 字段实现动态类实例化：
 
```python
config = {
    'model': {
        '_target_': 'src.lib.models.MLP',
        'input_dim': 784,
        'hidden_dims': [256, 128],
        'output_dim': 10
    }
}
model = instantiate(config['model'])  # 自动实例化 MLP
```
 
### 3.5 设备自动检测
 
Common 框架自动检测可用硬件，无需手动配置。
 
#### 核心类：DeviceManager
 
位于 `src/utils/device_management.py`
 
```python
from src.utils import get_device_manager
 
# 获取设备管理器（单例模式）
device_manager = get_device_manager()
 
# 自动获取当前设备（优先级：NPU > GPU > CPU）
device = device_manager.get_current_device()
 
# 获取设备类型
device_type = device_manager.get_device_type()  # 'npu' / 'gpu' / 'cpu'
```
 
#### 支持的硬件优先级
 
```
NPU (华为昇腾) > GPU (NVIDIA CUDA) > CPU (中央处理器)
```
 
#### 训练模式
 
```python
from src.utils.device_management import TrainingMode
 
# 自动检测（推荐）
TrainingMode.AUTO
 
# 单设备训练
TrainingMode.SINGLE
 
# 单机多卡训练
TrainingMode.SINGLE_MACHINE_MULTI_DEVICE
 
# 多机多卡训练
TrainingMode.MULTI_MACHINE_MULTI_DEVICE
```
 
### 3.6 注册管理与扩展机制
 
Common 框架通过注册表系统实现组件的动态注册和构建。
 
#### 核心类：Registry
 
位于 `src/utils/registry.py`
 
```python
from src.utils import Registry
 
# 创建注册表
MODELS = Registry('models')
 
# 注册组件
@MODELS.register('mlp')
class MLP(BaseModel):
    def __init__(self, input_dim, hidden_dims, output_dim):
        ...
 
# 构建组件
model = MODELS.build({'type': 'mlp', 'input_dim': 784, 'output_dim': 10})
 
# 获取注册表
model_cls = MODELS.get('mlp')
 
# 列出所有注册的组件
MODELS.list_registered()
```
 
#### 内置注册表
 
| 注册表名 | 组件类型 | 示例组件 |
|---------|---------|----------|
| MODELS | 模型 | MLP, CNN, Transformer |
| LOSSES | 损失函数 | CrossEntropyLoss, MSELoss |
| OPTIMIZERS | 优化器 | Adam, SGD, AdamW |
| EVALUATORS | 评估器 | AccuracyEvaluator, F1Evaluator |
| DATA_FETCHERS | 数据获取器 | CSVDataFetcher, JSONDataFetcher |
| DATA_PROCESSORS | 数据处理器 | StandardScaler, MinMaxScaler |
| FEATURE_CONSTRUCTORS | 特征构建器 | PolynomialFeatures |
| FEATURE_SELECTORS | 特征选择器 | VarianceThreshold |
| IMBALANCE_HANDLERS | 不平衡处理器 | RandomOverSampler |
| ALGORITHMS | 算法 | Supervised, Unsupervised, RL |
| HOOKS | 训练钩子 | CheckpointHook, EarlyStoppingHook |
 
### 3.7 分布式引擎
 
Common 框架原生支持分布式数据并行（DDP），适配多种通信后端。
 
#### 分布式训练流程 (DDP)
 
```
┌─────────────────────────────────────────┐
│  DistributedEngine.initialize()        │
│  自动检测设备，设置分布式环境            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  prepare_dataloader(train_loader)      │
│                                    │
    ┌───────────────────────────────┐   │
    │ 如果是 Dataset:                  │
    │   包装为 DatasetWrapper           │
    │                               │
    │ 如果是分布式环境:                 │
    │   添加 DistributedSampler          │
    └───────────────────────────────┘   │
                   │
                   ▼
┌─────────────────────────────────────────┐
│  每个进程只处理部分数据                  │
│  梯度同步 via all_reduce                 │
└─────────────────────────────────────────┘
```
 
#### 支持的分布式后端
 
| 后端 | 适用场景 |
|-----|---------|
| NCCL | NVIDIA GPU 分布式训练 |
| GLOO | CPU/跨机器分布式训练 |
| HCCL | 华为 NPU 分布式训练 |
 
---
 
## 四、配置文件编写
 
### 4.1 Trainer 配置结构
 
Trainer 从配置文件中读取以下内容：
 
```yaml
# ============================================================
# 数据配置 (data)
# ============================================================
data:
  # 数据获取器配置
  fetcher:
    _target_: "src.lib.data_fetching.CSVDataFetcher"  # 动态实例化
    source: "data.csv"          # 数据源路径
    target_column: "label"     # 目标列名
 
  # 数据处理器列表（按顺序执行）
  processors:
    - _target_: "src.lib.data_processing.StandardScaler"
    - _target_: "src.lib.data_processing.MinMaxScaler"
 
  # 特征构建器
  constructors:
    - _target_: "src.lib.feature_construction.PolynomialFeatures"
      degree: 2
 
  # 特征选择器
  selectors:
    - _target_: "src.lib.feature_selection.VarianceThreshold"
 
  # 不平衡处理
  imbalance_handler:
    _target_: "src.lib.imbalance_handling.RandomOverSampler"
 
# ============================================================
# 数据集划分配置 (dataset)
# ============================================================
dataset:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  shuffle: true
  random_seed: 42
 
# ============================================================
# 模型配置 (model)
# ============================================================
model:
  _target_: "src.lib.models.MLP"
  input_dim: 784              # 输入维度
  hidden_dims: [256, 128]    # 隐藏层维度列表
  output_dim: 10              # 输出维度
  activation: "relu"         # 激活函数: relu/sigmoid/tanh
  dropout: 0.5               # Dropout 比率
 
# ============================================================
# 损失函数配置 (loss)
# ============================================================
loss:
  _target_: "src.lib.loss_func.CrossEntropyLoss"
 
# ============================================================
# 优化器配置 (optimizer)
# ============================================================
optimizer:
  _target_: "src.lib.optimizer.Adam"
  lr: 0.001                  # 学习率
  betas: [0.9, 0.999]        # 动量参数
  eps: 1e-8                  # 数值稳定参数
  weight_decay: 0.0          # 权重衰减
 
# ============================================================
# 评估器配置 (evaluator)
# ============================================================
evaluator:
  _target_: "src.lib.evaluator.AccuracyEvaluator"
 
# ============================================================
# 算法配置 (algorithm) - 默认supervised
# ============================================================
algorithm:
  type: "supervised"         # supervised / unsupervised / dqn / ppo / sac
 
# ============================================================
# 训练配置 (training)
# ============================================================
training:
  epochs: 10                 # 训练轮数（监督学习）
  batch_size: 32             # 批次大小
 
# ============================================================
# 环境配置 (environment) - 强化学习用
# ============================================================
environment:
  _target_: "src.training.algorithm.rl.GymEnvWrapper"
  env_name: "CartPole-v1"
 
# ============================================================
# 分布式配置 (distributed)
# ============================================================
distributed:
  mode: "auto"               # 训练模式: auto/single/single_multi/multi_multi
  backend: "auto"            # 通信后端: auto/nccl/gloo/hccl
  auto_launch: true         # 是否自动启动分布式进程
 
# ============================================================
# 钩子配置 (hooks)
# ============================================================
hooks:
  checkpoint:
    _target_: "src.training.hook.checkpoint.CheckpointHook"
    save_dir: "checkpoints/"
    save_interval: 1
    save_best: true
    monitor: "val_accuracy"
    mode: "max"
 
  early_stopping:
    _target_: "src.training.hook.early_stopping.EarlyStoppingHook"
    patience: 5
    monitor: "val_loss"
    mode: "min"
 
# ============================================================
# 日志配置 (logging)
# ============================================================
logging:
  level: "INFO"
  log_dir: "./logs"
  console_output: true
  file_output: true
 
# ============================================================
# 自定义组件导入 (custom_imports)
# ============================================================
custom_imports:
  - "my_models"
```
 
### 4.2 最小配置示例
 
```yaml
# 最小配置 - 只需指定核心组件
data:
  fetcher:
    _target_: "src.lib.data_fetching.CSVDataFetcher"
    source: "data.csv"
    target_column: "label"
 
model:
  _target_: "src.lib.models.MLP"
  input_dim: 784
  output_dim: 10
 
loss:
  _target_: "src.lib.loss_func.CrossEntropyLoss"
 
optimizer:
  _target_: "src.lib.optimizer.Adam"
  lr: 0.001
 
training:
  epochs: 10
  batch_size: 32
```
 
### 4.3 环境变量覆盖
 
配置支持通过环境变量覆盖：
 
```bash
# 设置环境变量
export ML_TRAINING_EPOCHS=50
export ML_TRAINING_BATCH_SIZE=64
export ML_OPTIMIZER_LR=0.0001
export ML_MODEL_HIDDEN_DIMS="[512,256]"
```
 
---
 
## 五、使用场景示例
 
### 5.1 配置驱动的默认训练（Trainer 入口）
 
最简单的训练流程，只需准备数据和配置文件。
 
#### 1. 准备数据文件 (data.csv)
 
```csv
feature_1,feature_2,feature_3,label
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```
 
#### 2. 创建配置文件 (config.yaml)
 
```yaml
data:
  fetcher:
    _target_: "src.lib.data_fetching.CSVDataFetcher"
    source: "data.csv"
    target_column: "label"
 
  dataset:
    train_ratio: 0.8
    val_ratio: 0.2
 
model:
  _target_: "src.lib.models.MLP"
  input_dim: 3
  hidden_dims: [16, 8]
  output_dim: 2
  activation: "relu"
 
loss:
  _target_: "src.lib.loss_func.CrossEntropyLoss"
 
optimizer:
  _target_: "src.lib.optimizer.Adam"
  lr: 0.001
 
evaluator:
  _target_: "src.lib.evaluator.AccuracyEvaluator"
 
training:
  epochs: 10
  batch_size: 32
```
 
#### 3. 运行训练（使用 Trainer 统一入口）
 
```python
from src.training import Trainer
 
# 创建 Trainer 并训练
trainer = Trainer('config.yaml')
history = trainer.train()
```
 
#### 4. 或者使用一行代码
 
```python
from src.training import train
 
history = train('config.yaml')
```
 
### 5.2 个性化扩展的训练
 
自定义组件并在配置中使用。
 
#### 1. 注册自定义模型
 
```python
from src.lib.models import MODELS, BaseModel
 
@MODELS.register('custom_mlp')
class CustomMLP(BaseModel):
    """带批量归一化的自定义MLP"""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
 
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
 
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)
```
 
#### 2. 使用自定义模型
 
```yaml
model:
  type: "custom_mlp"  # 使用注册名
  input_dim: 784
  hidden_dims: [512, 256, 128]
  output_dim: 10
```
 
#### 3. 注册自定义模块
 
```yaml
custom_imports:
  - "my_custom_models"  # 模块会注册到各个 Registry
```
 
### 5.3 分布式训练
 
多卡分布式训练配置。
 
#### 1. 分布式配置文件
 
```yaml
data:
  fetcher:
    _target_: "src.lib.data_fetching.CSVDataFetcher"
    source: "data.csv"
    target_column: "label"
 
model:
  _target_: "src.lib.models.MLP"
  input_dim: 784
  hidden_dims: [256, 128]
  output_dim: 10
 
loss:
  _target_: "src.lib.loss_func.CrossEntropyLoss"
 
optimizer:
  _target_: "src.lib.optimizer.Adam"
  lr: 0.001
 
training:
  epochs: 10
  batch_size: 64
 
distributed:
  mode: "auto"
  backend: "auto"
  auto_launch: true
```
 
#### 2. 运行分布式训练
 
```bash
# Trainer 会自动检测并启动分布式训练
python train.py
 
# 或者手动指定
python -m torch.distributed.launch --nproc_per_node=4 train.py
```
 
#### 3. Trainer 自动处理
 
```python
from src.training import Trainer
 
# Trainer 会自动检测分布式环境
trainer = Trainer('config.yaml')
 
# 自动进行分布式训练
history = trainer.train()
```
 
### 5.4 检查点重训
 
从保存的检查点恢复训练。
 
#### 1. 配置检查点
 
```yaml
hooks:
  checkpoint:
    _target_: "src.training.hook.checkpoint.CheckpointHook"
    save_dir: "checkpoints/"
    save_interval: 1
    save_best: true
    monitor: "val_accuracy"
    mode: "max"
```
 
#### 2. 训练时自动保存
 
```python
from src.training import Trainer
 
trainer = Trainer('config.yaml')
history = trainer.train()
 
# 检查点自动保存到 checkpoints/
```
 
#### 3. 从检查点恢复训练
 
```python
from src.training import Trainer
 
trainer = Trainer('config.yaml')
 
# 加载检查点
trainer.load('checkpoints/best_model.pth')
 
# 继续训练
history = trainer.train()
```
 
#### 4. 评估或预测
 
```python
# 评估
metrics = trainer.evaluate()
 
# 预测
predictions = trainer.predict(test_data)
```
 
### 5.5 自定义强化学习
 
使用框架的强化学习模块训练 RL 算法。
 
#### 1. 强化学习配置
 
```yaml
algorithm:
  type: "dqn"
 
environment:
  _target_: "src.training.algorithm.rl.GymEnvWrapper"
  env_name: "CartPole-v1"
 
model:
  _target_: "src.training.algorithm.rl.DQNNetwork"
  state_dim: 4
  action_dim: 2
  hidden_dims: [128, 64]
 
loss:
  _target_: "src.lib.loss_func.MSELoss"
 
optimizer:
  _target_: "src.lib.optimizer.Adam"
  lr: 0.001
 
training:
  total_steps: 10000
  batch_size: 64
  gamma: 0.99
 
distributed:
  mode: "single"  # RL 通常不使用分布式
  auto_launch: false
```
 
#### 2. 运行强化学习训练（使用 Trainer）
 
```python
from src.training import Trainer
 
trainer = Trainer('rl_config.yaml')
 
# Trainer 会自动检测 RL 算法并调用对应的训练流程
history = trainer.train()
```
 
#### 3. 支持的 RL 算法
 
框架内置以下 RL 算法（通过 ALGORITHMS 注册表）：
 
| 算法 | 注册名 | 适用场景 |
|-----|-------|----------|
| DQN | dqn | 离散动作空间 |
| PPO | ppo | 连续动作空间 |
| SAC | sac | 连续动作空间 |
| A2C | a2c | 离散/连续动作空间 |
 
---
 
## 六、快速开始
 
### 6.1 安装
 
```bash
pip install common # Not available now
```
 
### 6.2 最小示例
 
```python
from src.training import Trainer
 
# 一行代码启动训练
trainer = Trainer('config.yaml')
history = trainer.train()
```
 
### 6.3 完整示例
 
```python
from src.training import train
 
# 一行代码完成训练
history = train({
    'model': {'type': 'MLP', 'input_dim': 784, 'output_dim': 10},
    'data': {'fetcher': {'type': 'CSVDataFetcher', 'source': 'data.csv', 'target_column': 'label'}},
    'training': {'epochs': 10, 'batch_size': 32}
})
```
 
---
 
## 七、API 参考
 
### 7.1 核心类
 
| 类名 | 模块 | 功能 |
|-----|------|------|
| **Trainer** | src.training | 统一训练入口（Facade模式） |
| train() | src.training | 一行代码训练函数 |
| Config | src.utils | 配置管理 |
| DeviceManager | src.utils | 设备管理 |
| Registry | src.utils | 注册表 |
| DistributedEngine | src.training.distributed | 分布式引擎 |
 
### 7.2 Trainer 接口
 
```python
# 创建 Trainer
trainer = Trainer(config)  # config 可以是文件路径、字典或 Config 对象
 
# 训练
history = trainer.train(train_data, val_data)
 
# 评估
metrics = trainer.evaluate(test_data)
 
# 预测
predictions = trainer.predict(inputs)
 
# 保存/加载
trainer.save(filepath)
trainer.load(filepath)
 
# 获取配置
config = trainer.get_config()
```