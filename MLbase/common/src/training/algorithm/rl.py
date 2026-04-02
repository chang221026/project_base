"""Reinforcement learning algorithms.
 
Provides RL algorithm implementations supporting:
- Gym/Gymnasium standard interface
- Custom environment interface
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
"""
from typing import Any, Dict, Optional, List, Tuple
from collections import deque
import random
 
from .base import BaseAlgorithm, eval_mode
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
# === Replay Buffer ===
 
class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
 
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
 
        Args:
            capacity: Maximum buffer size.
        """
        self.buffer = deque(maxlen=capacity)
 
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer.
 
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
 
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch from buffer.
 
        Args:
            batch_size: Number of experiences to sample.
 
        Returns:
            Batch of experiences.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
 
    def __len__(self):
        """Get buffer size."""
        return len(self.buffer)
 
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
 
 
class RolloutBuffer:
    """Rollout buffer for on-policy algorithms (PPO, A2C)."""
 
    def __init__(self):
        """Initialize rollout buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
 
    def push(self, state, action, reward, value, log_prob, done):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
 
    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
 
    def __len__(self):
        return len(self.states)
 
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch from buffer.
 
        Args:
            batch_size: Number of experiences to sample.
 
        Returns:
            Batch of experiences.
        """
        if len(self.states) == 0:
            return []
 
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        return [
            (self.states[i], self.actions[i], self.rewards[i],
             self.values[i], self.log_probs[i], self.dones[i])
            for i in indices
        ]
 
 
# === Environment Wrappers ===
 
class GymWrapper:
    """Gym/Gymnasium environment adapter."""
 
    def __init__(self, config: Dict[str, Any]):
        """Initialize gym wrapper.
 
        Args:
            config: Configuration with:
                - name: Environment name (e.g., 'CartPole-v1')
                - render: Whether to render
        """
        self.env_name = config.get('name', 'CartPole-v1')
        self.render = config.get('render', False)
        self._env = None
 
    def create_env(self):
        """Create and return gym environment."""
        try:
            import gymnasium as gym
        except ImportError:
            import gym
 
        render_mode = 'human' if self.render else None
        self._env = gym.make(self.env_name, render_mode=render_mode)
        return self._env
 
    @property
    def observation_space(self):
        return self._env.observation_space if self._env else None
 
    @property
    def action_space(self):
        return self._env.action_space if self._env else None
 
 
class CustomEnvWrapper:
    """Custom environment adapter."""
 
    def __init__(self, config: Dict[str, Any]):
        """Initialize custom wrapper.
 
        Args:
            config: Configuration with:
                - class: Environment class path or _target_ specification
        """
        self.env_config = config
 
    def create_env(self):
        """Create and return custom environment."""
        from utils.config_management import instantiate
        return instantiate(self.env_config)
 
 
# === RL Base Algorithm ===
 
class RLAlgorithm(BaseAlgorithm):
    """Base reinforcement learning algorithm.
 
    Supports:
    - Gym/Gymnasium standard interface
    - Custom environment interface
    - Step-based training (instead of epoch-based)
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RL algorithm.
 
        Args:
            config: Configuration containing:
                - gamma: Discount factor
                - buffer_size: Replay buffer size
                - batch_size: Training batch size
                - environment: Environment configuration
        """
        super().__init__(config)
        self.gamma = self.config.get('gamma', 0.99)
        self.batch_size = self.config.get('batch_size', 64)
        self.buffer = ReplayBuffer(self.config.get('buffer_size', 100000))
        self.device = None
        self.env_wrapper = None
 
    def setup(self) -> None:
        """Setup components and device."""
        self.device = self._get_device()  # Already has CPU fallback in base class
        self._setup_env_wrapper()
 
    def _setup_env_wrapper(self) -> None:
        """Setup environment wrapper based on config."""
        env_config = self.config.get('environment', {})
 
        # 如果没有环境配置，不创建默认环境
        # 用户可能通过 train(env=...) 传递环境实例
        if not env_config:
            self.env_wrapper = None
            return
 
        env_type = env_config.get('type', None)
 
        # 如果有 _target_ 字段但没有 type，默认使用 custom
        if env_type is None:
            if '_target_' in env_config:
                env_type = 'custom'
            else:
                env_type = 'gym'
 
        if env_type == 'gym':
            self.env_wrapper = GymWrapper(env_config)
        else:
            self.env_wrapper = CustomEnvWrapper(env_config)
 
    def select_action(self, state: Any, is_eval: bool = False) -> Any:
        """Select action given state.
 
        Args:
            state: Current state.
            is_eval: Whether in evaluation mode.
 
        Returns:
            Selected action.
        """
        raise NotImplementedError("select_action must be implemented by subclass")
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step.
 
        Args:
            batch: Batch of experiences (or None to sample from buffer).
 
        Returns:
            Dictionary of losses.
        """
        raise NotImplementedError("train_step must be implemented by subclass")
 
    def val_step(self, batch: Any) -> Dict[str, float]:
        """Validation step - not typically used in RL."""
        return {}
 
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate policy.
 
        Args:
            env: Environment to evaluate in.
            num_episodes: Number of episodes to run.
 
        Returns:
            Evaluation metrics.
        """
        episode_rewards = []
 
        for _ in range(num_episodes):
            state = self._env_reset(env)
            total_reward = 0
            done = False
 
            while not done:
                with eval_mode(self.model) if hasattr(self, 'model') else eval_mode(self.actor):
                    action = self.select_action(state, is_eval=True)
                state, reward, terminated, truncated = self._env_step(env, action)
                done = terminated or truncated
                total_reward += reward
 
            episode_rewards.append(total_reward)
 
        return {
            'mean_reward': sum(episode_rewards) / len(episode_rewards),
            'max_reward': max(episode_rewards),
            'min_reward': min(episode_rewards),
        }
 
    def _env_reset(self, env):
        """Reset environment with Gym/Gymnasium compatibility.
 
        Args:
            env: Environment instance.
 
        Returns:
            Initial observation.
        """
        result = env.reset()
        if isinstance(result, tuple):
            return result[0]  # (observation, info)
        return result  # observation only (old gym)
 
    def _env_step(self, env, action):
        """Step environment with Gym/Gymnasium compatibility.
 
        Args:
            env: Environment instance.
            action: Action to take.
 
        Returns:
            Tuple of (next_state, reward, terminated, truncated).
        """
        result = env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, _ = result
        else:
            next_state, reward, done, _ = result
            terminated = done
            truncated = False
        return next_state, reward, terminated, truncated
 
    def fit(self, env=None, total_steps: int = 100000, eval_freq: int = 5000,
            eval_episodes: int = 10) -> Dict[str, Any]:
        """Train the agent (step-based, not epoch-based).
 
        Args:
            env: Environment instance. If None, creates from config.
            total_steps: Total training steps.
            eval_freq: Evaluation frequency (in steps).
            eval_episodes: Number of episodes for evaluation.
 
        Returns:
            Training history.
        """
        if env is None:
            if self.env_wrapper is None:
                raise ValueError(
                    "No environment provided. Either pass 'env' to train() "
                    "or configure 'environment' in config."
                )
            env = self.env_wrapper.create_env()
 
        logger.info(f"Starting RL training for {total_steps} steps")
 
        state = self._env_reset(env)
        episode_reward = 0
        episode_length = 0
        history = {'train': [], 'eval': []}
        last_eval_step = 0
 
        for step in range(total_steps):
            # Select and execute action
            action = self.select_action(state)
            # Handle action tuple from some algorithms (e.g., PPO returns (action, log_prob))
            env_action = action[0] if isinstance(action, tuple) else action
            next_state, reward, terminated, truncated = self._env_step(env, env_action)
            done = terminated or truncated
 
            # Store transition
            self.store_transition(state, action, reward, next_state, done)
 
            episode_reward += reward
            episode_length += 1
            state = next_state
 
            # Train if enough samples
            if len(self.buffer) >= self.batch_size:
                metrics = self.train_step(None)
                self._global_step += 1
 
            # Episode end
            if done:
                state = self._env_reset(env)
 
                # Evaluate when enough steps have passed since last evaluation
                if step - last_eval_step >= eval_freq and step > 0:
                    eval_metrics = self.evaluate(env, eval_episodes)
                    history['eval'].append({
                        'step': step,
                        **eval_metrics,
                        'episode_reward': episode_reward,
                    })
                    last_eval_step = step
                    logger.info(
                        f"Step {step}: reward={episode_reward:.2f}, "
                        f"mean_eval={eval_metrics['mean_reward']:.2f}"
                    )
                else:
                    history['train'].append({
                        'step': step,
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                    })
 
                episode_reward = 0
                episode_length = 0
 
        self._trained = True
        logger.info("RL training completed")
        return history
 
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer.
 
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.buffer.push(state, action, reward, next_state, done)
 
 
# === PPO Algorithm ===
 
class PPO(RLAlgorithm):
    """Proximal Policy Optimization algorithm.
 
    An on-policy actor-critic algorithm with clipped objective.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PPO.
 
        Args:
            config: Configuration including:
                - clip_ratio: PPO clipping parameter (default: 0.2)
                - value_coef: Value loss coefficient (default: 0.5)
                - entropy_coef: Entropy bonus coefficient (default: 0.01)
                - gae_lambda: GAE lambda parameter (default: 0.95)
                - n_epochs: Number of update epochs per rollout (default: 10)
        """
        super().__init__(config)
        self.clip_ratio = self.config.get('clip_ratio', 0.2)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.n_epochs = self.config.get('n_epochs', 10)
        self.buffer = RolloutBuffer()
 
    def setup(self) -> None:
        """Setup actor and critic networks."""
        super().setup()
 
        # Build actor (policy network)
        self.actor = self._build_model(self.config.get('actor', {}))
        self.actor = self.actor.to(str(self.device))
        self.model = self.actor  # For compatibility with base class
 
        # Build critic (value network)
        self.critic = self._build_model(self.config.get('critic', {}))
        self.critic = self.critic.to(str(self.device))
 
        # Build optimizer
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = self._build_optimizer(params, self.config.get('optimizer'))
 
        logger.info(f"PPO setup: actor={type(self.actor).__name__}, critic={type(self.critic).__name__}")
 
    def select_action(self, state: Any, is_eval: bool = False) -> Any:
        """Select action using policy network.
 
        Args:
            state: Current state.
            is_eval: Whether in evaluation mode.
 
        Returns:
            Selected action (and log_prob during training).
        """
        import torch
 
        state_tensor = self._move_to_device(state)
 
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
 
            if is_eval:
                # Greedy action selection
                action = action_logits.argmax(dim=-1)
                return action.item() if action.dim() == 0 else action
 
            # Sample action from distribution
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
 
            return action.item() if action.dim() == 0 else action, log_prob
 
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation.
 
        Args:
            rewards: List of rewards.
            values: List of value estimates.
            dones: List of done flags.
            next_value: Value estimate for next state.
 
        Returns:
            Tuple of (advantages, returns).
        """
        advantages = []
        gae = 0
 
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
 
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
 
        advantages = self._move_to_device(advantages)
        returns = advantages + self._move_to_device(values)
 
        return advantages, returns
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """PPO training step.
 
        Args:
            batch: Ignored, uses rollout buffer.
 
        Returns:
            Dictionary of losses.
        """
        import torch
 
        if len(self.buffer) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
 
        # Get buffer data
        states = self._move_to_device(self.buffer.states)
        actions = self._move_to_device(self.buffer.actions, keep_dtype=True)
        old_log_probs = self._move_to_device(self.buffer.log_probs)
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
 
        # Compute next value for GAE
        with torch.no_grad():
            next_value = self.critic(states[-1]).item()
 
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
 
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
 
        # PPO update epochs
        for _ in range(self.n_epochs):
            # Get action logits
            action_logits = self.actor(states)
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
 
            # New log probs
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
 
            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
 
            # Value loss
            value_pred = self.critic(states).squeeze()
            value_loss = ((value_pred - returns) ** 2).mean()
 
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
 
        # Clear buffer after update
        self.buffer.clear()
 
        return {
            'policy_loss': total_policy_loss / self.n_epochs,
            'value_loss': total_value_loss / self.n_epochs,
            'entropy': total_entropy / self.n_epochs,
        }
 
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in rollout buffer.
 
        For PPO, we need to store state, action, reward, value, log_prob, done.
        """
        import torch
 
        # Handle action that may include log_prob from select_action
        if isinstance(action, tuple):
            action, log_prob_from_select = action
        else:
            log_prob_from_select = None
 
        state_tensor = self._move_to_device(state)
 
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if log_prob_from_select is not None:
                log_prob = log_prob_from_select
            else:
                log_prob = dist.log_prob(torch.tensor(action, device=str(self.device)))
            value = self.critic(state_tensor).item()
 
        self.buffer.push(state, action, reward, value, log_prob.item() if hasattr(log_prob, 'item') else log_prob, done)
 
    def save(self, filepath: str) -> None:
        """Save PPO state."""
        import torch
 
        state = {
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self._global_step,
        }
        torch.save(state, filepath)
        logger.info(f"PPO saved to {filepath}")
 
    def load(self, filepath: str) -> None:
        """Load PPO state."""
        import torch
 
        device_str = str(self.device)
        state = torch.load(filepath, map_location=device_str, weights_only=False)
 
        self.actor.load_state_dict(state['actor_state'])
        self.critic.load_state_dict(state['critic_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self._global_step = state.get('global_step', 0)
 
        logger.info(f"PPO loaded from {filepath}")
 
 
# === SAC Algorithm ===
 
class SAC(RLAlgorithm):
    """Soft Actor-Critic algorithm.
 
    An off-policy actor-critic algorithm with entropy regularization.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SAC.
 
        Args:
            config: Configuration including:
                - tau: Soft update coefficient (default: 0.005)
                - alpha: Entropy coefficient (default: 0.2)
                - auto_entropy: Whether to auto-tune alpha (default: True)
        """
        super().__init__(config)
        self.tau = self.config.get('tau', 0.005)
        self.alpha = self.config.get('alpha', 0.2)
        self.auto_entropy = self.config.get('auto_entropy', True)
        self.buffer = ReplayBuffer(self.config.get('buffer_size', 100000))
        self.target_entropy = None
        self.log_alpha = None
        self.alpha_optimizer = None
 
    def setup(self) -> None:
        """Setup actor, critic, and target networks."""
        super().setup()
 
        # Build actor
        self.actor = self._build_model(self.config.get('actor', {}))
        self.actor = self.actor.to(str(self.device))
        self.model = self.actor  # For compatibility
 
        # Build Q-networks
        q_config = self.config.get('q_network', {})
        self.q1 = self._build_model(q_config).to(str(self.device))
        self.q2 = self._build_model(q_config).to(str(self.device))
 
        # Build target Q-networks
        self.target_q1 = self._build_model(q_config).to(str(self.device))
        self.target_q2 = self._build_model(q_config).to(str(self.device))
 
        # Initialize target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
 
        # Build optimizers
        self.actor_optimizer = self._build_optimizer(
            self.actor, self.config.get('actor_optimizer')
        )
        self.critic_optimizer = self._build_optimizer(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.get('critic_optimizer')
        )
 
        # Auto entropy tuning
        if self.auto_entropy:
            import torch
            self.log_alpha = torch.zeros(1, requires_grad=True, device=str(self.device))
            self.alpha_optimizer = self._build_optimizer(
                [self.log_alpha], {'type': 'Adam', 'lr': 3e-4}
            )
            # Target entropy = -dim(action_space)
            if hasattr(self.env_wrapper, 'action_space'):
                self.target_entropy = -self.env_wrapper.action_space.shape[0]
 
        logger.info(f"SAC setup: actor={type(self.actor).__name__}")
 
    def select_action(self, state: Any, is_eval: bool = False) -> Any:
        """Select action using policy network.
 
        Args:
            state: Current state.
            is_eval: Whether in evaluation mode.
 
        Returns:
            Selected action.
        """
        import torch
 
        state_tensor = self._move_to_device(state)
 
        with torch.no_grad():
            if is_eval:
                # Deterministic action for evaluation
                action = self.actor(state_tensor)
                if isinstance(action, tuple):
                    action = action[0]  # mean action
            else:
                # Sample action for training
                action = self.actor(state_tensor)
                if isinstance(action, tuple):
                    action, _ = action  # sampled action and log_prob
 
        return action.cpu().numpy() if hasattr(action, 'cpu') else action
 
    def soft_update(self, target, source):
        """Soft update target network.
 
        target = tau * source + (1 - tau) * target
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """SAC training step.
 
        Args:
            batch: None to sample from buffer.
 
        Returns:
            Dictionary of losses.
        """
        import torch
 
        if len(self.buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'q_loss': 0.0, 'alpha': self.alpha}
 
        # Sample batch
        experiences = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
 
        states = self._move_to_device(states)
        actions = self._move_to_device(actions, keep_dtype=True)
        rewards = self._move_to_device(rewards)
        next_states = self._move_to_device(next_states)
        dones = self._move_to_device(dones)
 
        # Get current alpha
        alpha = self.log_alpha.exp() if self.log_alpha is not None else self.alpha
 
        # === Update critic ===
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next = self.target_q1(next_states, next_actions)
            q2_next = self.target_q2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next
 
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = ((q1_pred - target_q) ** 2).mean()
        q2_loss = ((q2_pred - target_q) ** 2).mean()
        q_loss = q1_loss + q2_loss
 
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
 
        # === Update actor ===
        new_actions, log_probs = self.actor(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
 
        actor_loss = (alpha * log_probs - q_new).mean()
 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
 
        # === Update alpha ===
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy and self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
 
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
 
            self.alpha = self.log_alpha.exp().item()
 
        # === Soft update target networks ===
        self.soft_update(self.target_q1, self.q1)
        self.soft_update(self.target_q2, self.q2)
 
        return {
            'actor_loss': actor_loss.item(),
            'q_loss': q_loss.item(),
            'alpha': self.alpha,
        }
 
    def save(self, filepath: str) -> None:
        """Save SAC state."""
        import torch
 
        state = {
            'actor_state': self.actor.state_dict(),
            'q1_state': self.q1.state_dict(),
            'q2_state': self.q2.state_dict(),
            'target_q1_state': self.target_q1.state_dict(),
            'target_q2_state': self.target_q2.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'global_step': self._global_step,
        }
        if self.log_alpha is not None:
            state['log_alpha'] = self.log_alpha
            state['alpha_optimizer_state'] = self.alpha_optimizer.state_dict()
 
        torch.save(state, filepath)
        logger.info(f"SAC saved to {filepath}")
 
    def load(self, filepath: str) -> None:
        """Load SAC state."""
        import torch
 
        device_str = str(self.device)
        state = torch.load(filepath, map_location=device_str, weights_only=False)
 
        self.actor.load_state_dict(state['actor_state'])
        self.q1.load_state_dict(state['q1_state'])
        self.q2.load_state_dict(state['q2_state'])
        self.target_q1.load_state_dict(state['target_q1_state'])
        self.target_q2.load_state_dict(state['target_q2_state'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer_state'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer_state'])
 
        if 'log_alpha' in state:
            self.log_alpha = state['log_alpha'].to(device_str)
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer_state'])
            self.alpha = self.log_alpha.exp().item()
 
        self._global_step = state.get('global_step', 0)
        logger.info(f"SAC loaded from {filepath}")
 