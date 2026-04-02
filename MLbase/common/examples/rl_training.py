#!/usr/bin/env python
"""强化学习训练示例 - 展示框架的便利性与可扩展性。
 
展示:
1. 快速开始: DQN 训练 CartPole - 配置驱动的简洁用法
2. 自定义网络: PPO Actor-Critic - 注册自定义网络组件
3. 自定义环境: 简单网格世界 - 自定义环境接口规范
 
依赖说明:
- Demo 1, 2 需要 gymnasium 或 gym: pip install gymnasium
- Demo 3 仅依赖框架本身，无需额外安装
 
推荐用法: 使用 Trainer 作为统一入口点
"""
 
import sys
from pathlib import Path
 
# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
import torch
import torch.nn as nn
 
from training.trainer import Trainer
from lib.models import MODELS
 
# 抑制详细日志
import logging
logging.getLogger().setLevel(logging.WARNING)
 
 
def check_gym_available():
    """检查 gym/gymnasium 是否可用。"""
    try:
        import gymnasium
        return True
    except ImportError:
        try:
            import gym
            return True
        except ImportError:
            return False
 
 
GYM_AVAILABLE = check_gym_available()
 
 
# ============================================================================
# Demo 1: 快速开始 - PPO 训练 CartPole
# ============================================================================
 
def demo_quick_start_dqn():
    """展示框架便利性 - 几行代码完成 PPO 训练。
 
    CartPole-v1 环境:
    - 状态空间: 4 维连续 (位置、速度、角度、角速度)
    - 动作空间: 2 个离散动作 (左/右)
    - 目标: 保持杆子平衡，每步得 1 分
    """
    if not GYM_AVAILABLE:
        print("=" * 60)
        print("Demo 1: 快速开始 - PPO 训练 CartPole")
        print("=" * 60)
        print("\n   [跳过] 需要 gymnasium 或 gym 库")
        print("   安装命令: pip install gymnasium")
        return None
 
    print("=" * 60)
    print("Demo 1: 快速开始 - PPO 训练 CartPole")
    print("=" * 60)
 
    # -------------------------------------------------------------------------
    # 配置驱动的简洁用法 - 使用 Trainer 统一入口
    # -------------------------------------------------------------------------
    config = {
        'algorithm': {
            'type': 'ppo',
            'actor': {
                'type': 'MLP',
                'input_dim': 4,        # CartPole 状态维度
                'hidden_dims': [128, 64],
                'output_dim': 2        # 动作数量
            },
            'critic': {
                'type': 'MLP',
                'input_dim': 4,        # CartPole 状态维度
                'hidden_dims': [128, 64],
                'output_dim': 1        # 价值输出
            },
            'gamma': 0.99,             # 折扣因子
            'clip_ratio': 0.2,         # PPO 裁剪比率
            'gae_lambda': 0.95,        # GAE 参数
            'n_epochs': 10,            # 更新轮数
            'entropy_coef': 0.01,      # 熵系数
            'optimizer': {'type': 'Adam', 'lr': 0.001}
        },
        'environment': {
            'type': 'gym',
            'name': 'CartPole-v1'  # Gymnasium 标准环境
        },
        'training': {
            'total_steps': 3000,
            'eval_freq': 500,
            'eval_episodes': 5
        },
        'logging': {
            'level': 'WARNING',
            'console_output': False
        }
    }
 
    print("\n   环境: CartPole-v1")
    print("   算法: PPO")
    print("   网络: MLP (4 -> 128 -> 64 -> 2)")
 
    # -------------------------------------------------------------------------
    # 创建 Trainer 并训练 - 统一入口点
    # -------------------------------------------------------------------------
    print("\n   开始训练...")
 
    trainer = Trainer(config)
    history = trainer.train()
 
    # -------------------------------------------------------------------------
    # 打印训练结果
    # -------------------------------------------------------------------------
    print("\n   训练结果:")
    print("   " + "-" * 40)
    print(f"   {'Step':<10} {'Reward':<15} {'Eval Reward':<15}")
    print("   " + "-" * 40)
 
    for record in history['eval']:
        step = record.get('step', 0)
        ep_reward = record.get('episode_reward', 0)
        eval_reward = record.get('mean_reward', 0)
        print(f"   {step:<10} {ep_reward:<15.1f} {eval_reward:<15.1f}")
 
    final_reward = history['eval'][-1].get('mean_reward', 0) if history['eval'] else 0
    print("   " + "-" * 40)
    print(f"\n   最终评估奖励: {final_reward:.1f}")
 
    return trainer
 
 
# ============================================================================
# Demo 2: 自定义网络 - PPO Actor-Critic
# ============================================================================
 
@MODELS.register('PolicyNetwork')
class PolicyNetwork(nn.Module):
    """自定义策略网络 - 输出动作概率分布。
 
    展示如何注册自定义网络组件供框架使用。
    """
 
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
 
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)
 
 
@MODELS.register('ValueNetwork')
class ValueNetwork(nn.Module):
    """自定义价值网络 - 评估状态价值。
 
    PPO 需要独立的 Actor 和 Critic 网络。
    """
 
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
 
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)
 
 
def demo_custom_network_ppo():
    """展示框架可扩展性 - 使用自定义网络训练 PPO。
 
    PPO (Proximal Policy Optimization) 特点:
    - On-policy 算法
    - 使用 GAE (Generalized Advantage Estimation)
    - 策略裁剪保证训练稳定
    """
    if not GYM_AVAILABLE:
        print("\n" + "=" * 60)
        print("Demo 2: 自定义网络 - PPO Actor-Critic")
        print("=" * 60)
        print("\n   [跳过] 需要 gymnasium 或 gym 库")
        print("   安装命令: pip install gymnasium")
        return None
 
    print("\n" + "=" * 60)
    print("Demo 2: 自定义网络 - PPO Actor-Critic")
    print("=" * 60)
 
    # -------------------------------------------------------------------------
    # 使用自定义注册的网络组件 - 配置驱动
    # -------------------------------------------------------------------------
    config = {
        'algorithm': {
            'type': 'ppo',
            'actor': {
                'type': 'PolicyNetwork',  # 使用自定义注册的网络
                'input_dim': 4,
                'hidden_dim': 64,
                'output_dim': 2
            },
            'critic': {
                'type': 'ValueNetwork',   # 使用自定义注册的网络
                'input_dim': 4,
                'hidden_dim': 64
            },
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'n_epochs': 4,
            'batch_size': 64,
            'optimizer': {'type': 'Adam', 'lr': 0.0003}
        },
        'environment': {
            'type': 'gym',
            'name': 'CartPole-v1'
        },
        'training': {
            'total_steps': 3000,
            'eval_freq': 500,
            'eval_episodes': 5
        },
        'logging': {
            'level': 'WARNING',
            'console_output': False
        }
    }
 
    print("\n   环境: CartPole-v1")
    print("   算法: PPO")
    print("   Actor: PolicyNetwork (自定义)")
    print("   Critic: ValueNetwork (自定义)")
 
    # -------------------------------------------------------------------------
    # 训练 PPO - 使用 Trainer 统一入口
    # -------------------------------------------------------------------------
    print("\n   开始训练...")
 
    trainer = Trainer(config)
    history = trainer.train()
 
    # -------------------------------------------------------------------------
    # 打印训练结果
    # -------------------------------------------------------------------------
    print("\n   训练结果:")
    print("   " + "-" * 50)
    print(f"   {'Step':<10} {'Policy Loss':<15} {'Eval Reward':<15}")
    print("   " + "-" * 50)
 
    for record in history['eval']:
        step = record.get('step', 0)
        eval_reward = record.get('mean_reward', 0)
        print(f"   {step:<10} {'-':<15} {eval_reward:<15.1f}")
 
    final_reward = history['eval'][-1].get('mean_reward', 0) if history['eval'] else 0
    print("   " + "-" * 50)
    print(f"\n   最终评估奖励: {final_reward:.1f}")
 
    return trainer
 
 
# ============================================================================
# Demo 3: 自定义环境接口
# ============================================================================
 
import numpy as np
 
 
@MODELS.register('SimpleQNetwork')
class SimpleQNetwork(nn.Module):
    """简单的 Q 网络 - 不使用 BatchNorm，适合 RL。
 
    用于展示自定义网络组件。
    """
 
    def __init__(self, input_dim=25, hidden_dim=64, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
 
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # 确保 2D 输入
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)
 
 
class SimpleGridWorld:
    """简单网格世界环境 - 展示自定义环境接口规范。
 
    接口规范 (与 Gym 兼容):
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    - observation_space: 状态空间属性
    - action_space: 动作空间属性
    """
 
    def __init__(self, size=5):
        """初始化网格世界。
 
        Args:
            size: 网格大小 (size x size)
        """
        self.size = size
        self.agent_pos = 0
        self.goal_pos = size * size - 1  # 右下角
        self.max_steps = size * size
        self.steps = 0
 
        # 定义空间 (用于框架自动推断)
        self.observation_space = type('obj', (object,), {'shape': (4,)})()
        self.action_space = type('obj', (object,), {'n': 4})()  # 上下左右
 
    def _get_observation(self):
        """获取观察向量 - 返回 one-hot 编码的位置。"""
        obs = np.zeros(self.size * self.size, dtype=np.float32)
        obs[self.agent_pos] = 1.0
        return obs
 
    def reset(self):
        """重置环境。"""
        self.agent_pos = 0
        self.steps = 0
        return self._get_observation(), {}
 
    def step(self, action):
        """执行动作。
 
        Args:
            action: 0=上, 1=下, 2=左, 3=右
 
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        row, col = self.agent_pos // self.size, self.agent_pos % self.size
 
        # 执行动作
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 下
            row = min(self.size - 1, row + 1)
        elif action == 2:  # 左
            col = max(0, col - 1)
        elif action == 3:  # 右
            col = min(self.size - 1, col + 1)
 
        self.agent_pos = row * self.size + col
 
        # 计算奖励和终止条件
        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.steps >= self.max_steps)
 
        # 奖励: 到达目标 +1, 每步 -0.01
        reward = 1.0 if terminated else -0.01
 
        return self._get_observation(), reward, terminated, truncated, {'steps': self.steps}
 
 
def demo_custom_environment():
    """展示框架可扩展性 - 使用自定义环境训练 PPO。"""
    print("\n" + "=" * 60)
    print("Demo 3: 自定义环境 - 简单网格世界")
    print("=" * 60)
 
    # -------------------------------------------------------------------------
    # 创建自定义环境用于展示
    # -------------------------------------------------------------------------
    env = SimpleGridWorld(size=5)
 
    print(f"\n   环境: SimpleGridWorld (5x5)")
    print(f"   状态空间: {env.size * env.size} 维 one-hot 向量")
    print(f"   目标: 从左上角 (0,0) 移动到右下角 (4,4)")
    print(f"   动作: 上下左右 (4个)")
 
    # -------------------------------------------------------------------------
    # 配置并训练 - 使用简单的 Actor-Critic 网络，通过 Trainer 统一入口
    # -------------------------------------------------------------------------
    # 注意：自定义环境实例无法在分布式训练中序列化传递
    # 使用环境配置字典或禁用分布式启动
    config = {
        'algorithm': {
            'type': 'ppo',
            'actor': {
                'type': 'MLP',
                'input_dim': 25,           # 5x5 grid = 25
                'hidden_dims': [64],
                'output_dim': 4            # 4 actions
            },
            'critic': {
                'type': 'MLP',
                'input_dim': 25,           # 5x5 grid = 25
                'hidden_dims': [64],
                'output_dim': 1            # 1 value
            },
            'gamma': 0.95,
            'clip_ratio': 0.2,
            'gae_lambda': 0.95,
            'batch_size': 32,
            'buffer_size': 5000,
            'optimizer': {'type': 'Adam', 'lr': 0.01}
        },
        'training': {
            'total_steps': 5000,
            'eval_freq': 1000,
            'eval_episodes': 10
        },
        # 自定义环境实例无法在分布式场景下序列化
        # 使用环境配置或禁用分布式
        'environment': {
            '_target_': '__main__.SimpleGridWorld',
            'size': 5
        },
        'distributed': {
            'auto_launch': False  # 禁用自动分布式启动
        },
        'logging': {
            'level': 'WARNING',
            'console_output': False
        }
    }
 
    print("\n   开始训练...")
 
    trainer = Trainer(config)
    # 使用环境配置，不再传递实例
    history = trainer.train()
 
    print("\n   训练完成!")
 
    # -------------------------------------------------------------------------
    # 演示训练前后的策略对比
    # -------------------------------------------------------------------------
    print("\n   策略对比:")
 
    # 训练前的随机策略
    env_test = SimpleGridWorld(size=5)
    random_rewards = []
    for _ in range(10):
        state, _ = env_test.reset()
        total_reward = 0
        done = False
        while not done:
            import random
            action = random.randint(0, 3)
            state, reward, terminated, truncated, _ = env_test.step(action)
            total_reward += reward
            done = terminated or truncated
        random_rewards.append(total_reward)
 
    # 训练后的策略
    trained_rewards = []
    for _ in range(10):
        state, _ = env_test.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            with torch.no_grad():
                # 通过 TrainingFacade 获取算法实例
                algorithm = trainer.training_facade.get_algorithm()
                state_tensor = algorithm._move_to_device(state)
                q_values = algorithm.model(state_tensor)
                action = q_values.argmax().item()
            state, reward, terminated, truncated, _ = env_test.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        trained_rewards.append(total_reward)
 
    print("   " + "-" * 40)
    print(f"   {'策略':<15} {'平均奖励':<15} {'成功率':<10}")
    print("   " + "-" * 40)
    random_success = sum(1 for r in random_rewards if r > 0.5)
    trained_success = sum(1 for r in trained_rewards if r > 0.5)
    print(f"   {'随机策略':<15} {sum(random_rewards)/len(random_rewards):<15.2f} {random_success}/10")
    print(f"   {'训练后策略':<15} {sum(trained_rewards)/len(trained_rewards):<15.2f} {trained_success}/10")
    print("   " + "-" * 40)
 
    return trainer
 
 
# ============================================================================
# 主函数
# ============================================================================
 
def main():
    """运行所有示例。"""
    print("\n" + "=" * 60)
    print("强化学习训练示例")
    print("=" * 60)
    print("\n本示例展示框架的便利性与可扩展性:")
    print("  1. 快速开始: 几行代码完成 DQN 训练")
    print("  2. 自定义网络: 注册并使用自定义 Actor-Critic")
    print("  3. 自定义环境: 实现标准接口即可接入框架")
 
    # 运行示例
    demo_quick_start_dqn()
    demo_custom_network_ppo()
    demo_custom_environment()
 
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    print("\n框架特点:")
    print("  - 统一入口: Trainer 作为配置驱动的唯一入口点")
    print("  - 便利性: 配置驱动，自动设备检测，自动分布式")
    print("  - 可扩展性: 注册自定义组件，灵活的环境接口")
 
 
if __name__ == '__main__':
    main()