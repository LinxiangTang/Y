import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import yaml
import os
import logging
from datetime import datetime
import pandas as pd


@dataclass
class EnvConfig:
    grid_price_base: float
    grid_price_variance: float
    renewable_energy_base: float
    renewable_energy_variance: float
    data_center_max_load: float
    data_center_min_load: float
    energy_storage_capacity: float
    energy_storage_max_charge_rate: float
    energy_storage_max_discharge_rate: float
    energy_storage_efficiency: float
    demand_response_penalty: float
    carbon_emission_factor_grid: float
    carbon_emission_factor_renewable: float
    target_soc: float
    sla_threshold: float


@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    action_low: List[float]
    action_high: List[float]
    gamma: float
    tau: float
    policy_noise: float
    noise_clip: float
    policy_freq: int
    lr_actor: float
    lr_critic: float
    weight_decay: float


@dataclass
class TrainingConfig:
    episodes: int
    max_steps_per_episode: int
    batch_size: int
    buffer_capacity: int
    prioritized_replay: bool
    alpha: float
    beta: float
    beta_increment: float
    save_interval: int
    model_save_path: str
    log_path: str


@dataclass
class MetricsConfig:
    cost_weight: float
    carbon_weight: float
    sla_weight: float
    result_save_path: str


@dataclass
class Config:
    env: EnvConfig
    load_agent: AgentConfig
    storage_agent: AgentConfig
    training: TrainingConfig
    metrics: MetricsConfig


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    env_config = EnvConfig(**data['env'])
    load_agent_config = AgentConfig(**data['load_agent'])
    storage_agent_config = AgentConfig(**data['storage_agent'])
    training_config = TrainingConfig(**data['training'])
    metrics_config = MetricsConfig(**data['metrics'])
    return Config(env=env_config, load_agent=load_agent_config, storage_agent=storage_agent_config,
                  training=training_config, metrics=metrics_config)


class PowerDemandResponseEnv:
    def __init__(self, config: EnvConfig):
        self.grid_price_base = config.grid_price_base
        self.grid_price_var = config.grid_price_variance
        self.renewable_base = config.renewable_energy_base
        self.renewable_var = config.renewable_energy_variance
        self.dc_max_load = config.data_center_max_load
        self.dc_min_load = config.data_center_min_load
        self.es_capacity = config.energy_storage_capacity
        self.es_max_charge = config.energy_storage_max_charge_rate
        self.es_max_discharge = config.energy_storage_max_discharge_rate
        self.es_efficiency = config.energy_storage_efficiency
        self.dr_penalty = config.demand_response_penalty
        self.carbon_grid = config.carbon_emission_factor_grid
        self.carbon_renewable = config.carbon_emission_factor_renewable
        self.target_soc = config.target_soc
        self.sla_threshold = config.sla_threshold
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.grid_price = self.grid_price_base + np.random.normal(0, self.grid_price_var)
        self.renewable_output = self.renewable_base + np.random.normal(0, self.renewable_var)
        self.dc_load = np.random.uniform(self.dc_min_load, self.dc_max_load)
        self.es_soc = np.random.uniform(0.2, 0.8)
        self.es_charge = 0.0
        self.es_discharge = 0.0
        self.grid_power = 0.0
        self.renewable_power = 0.0
        self.cost_total = 0.0
        self.carbon_total = 0.0
        self.sla_violation = 0.0
        self.step_count = 0
        return self.get_state()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        load_agent_state = np.array([self.grid_price, self.renewable_output, self.dc_load, self.sla_violation])
        storage_agent_state = np.array([self.grid_price, self.es_soc, self.target_soc, self.renewable_output])
        return load_agent_state, storage_agent_state

    def compute_reward(self) -> float:
        cost_reward = -self.cost_total * self.metrics_config.cost_weight
        carbon_reward = -self.carbon_total * self.metrics_config.carbon_weight
        sla_reward = -self.sla_violation * self.metrics_config.sla_weight
        return cost_reward + carbon_reward + sla_reward

    def step(self, load_action: np.ndarray, storage_action: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        load_migration = load_action[0]
        storage_action_val = storage_action[0]
        self.dc_load = np.clip(self.dc_load + load_migration, self.dc_min_load, self.dc_max_load)
        if storage_action_val > 0:
            self.es_charge = np.clip(storage_action_val, 0, self.es_max_charge)
            self.es_discharge = 0.0
        else:
            self.es_discharge = np.clip(-storage_action_val, 0, self.es_max_discharge)
            self.es_charge = 0.0
        self.renewable_power = min(self.renewable_output, self.dc_load + self.es_charge - self.es_discharge)
        self.grid_power = self.dc_load + self.es_charge - self.es_discharge - self.renewable_power
        self.grid_power = max(0.0, self.grid_power)
        self.cost_total = self.grid_power * self.grid_price + self.es_charge * self.grid_price * self.es_efficiency
        self.carbon_total = self.grid_power * self.carbon_grid + self.renewable_power * self.carbon_renewable
        self.sla_violation = max(0.0, abs(self.dc_load - self.dc_max_load / 2) - self.sla_threshold) * self.dr_penalty
        self.es_soc += (self.es_charge * self.es_efficiency - self.es_discharge / self.es_efficiency) / self.es_capacity
        self.es_soc = np.clip(self.es_soc, 0.0, 1.0)
        self.grid_price = self.grid_price_base + np.random.normal(0, self.grid_price_var)
        self.renewable_output = self.renewable_base + np.random.normal(0, self.renewable_var)
        self.step_count += 1
        done = self.step_count >= self.training_config.max_steps_per_episode or self.es_soc <= 0.0 or self.es_soc >= 1.0
        reward = self.compute_reward()
        next_load_state, next_storage_state = self.get_state()
        info = {'cost': self.cost_total, 'carbon': self.carbon_total, 'sla': self.sla_violation, 'soc': self.es_soc}
        return (next_load_state, next_storage_state), reward, done, info


class MultiAgentReplayBuffer:
    def __init__(self, capacity: int, num_agents: int, state_dims: List[int], action_dims: List[int]):
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.states = [np.zeros((capacity, dim)) for dim in state_dims]
        self.actions = [np.zeros((capacity, dim)) for dim in action_dims]
        self.rewards = np.zeros(capacity)
        self.next_states = [np.zeros((capacity, dim)) for dim in state_dims]
        self.dones = np.zeros(capacity, dtype=bool)
        self.position = 0
        self.size = 0

    def add(self, states: List[np.ndarray], actions: List[np.ndarray], reward: float, next_states: List[np.ndarray],
            done: bool):
        for i in range(self.num_agents):
            self.states[i][self.position] = states[i]
            self.actions[i][self.position] = actions[i]
        self.rewards[self.position] = reward
        for i in range(self.num_agents):
            self.next_states[i][self.position] = next_states[i]
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = [torch.tensor(self.states[i][indices], dtype=torch.float32) for i in range(self.num_agents)]
        actions = [torch.tensor(self.actions[i][indices], dtype=torch.float32) for i in range(self.num_agents)]
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32)
        next_states = [torch.tensor(self.next_states[i][indices], dtype=torch.float32) for i in range(self.num_agents)]
        dones = torch.tensor(self.dones[indices], dtype=torch.bool)
        return states, actions, rewards, next_states, dones


class OUNoise:
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_low: float, action_high: float):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        action = self.action_low + (self.action_high - self.action_low) * (x + 1) / 2
        return action


class CriticNetwork(nn.Module):
    def __init__(self, state_dims: List[int], action_dims: List[int]):
        super().__init__()
        self.fc1 = nn.Linear(sum(state_dims) + sum(action_dims), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, states: List[torch.Tensor], actions: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(states + actions, dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MADDPGAgent:
    def __init__(self, agent_id: int, config: AgentConfig, state_dims: List[int], action_dims: List[int]):
        self.agent_id = agent_id
        self.config = config
        self.actor = ActorNetwork(config.state_dim, config.action_dim, config.action_low[0], config.action_high[0])
        self.target_actor = ActorNetwork(config.state_dim, config.action_dim, config.action_low[0],
                                         config.action_high[0])
        self.critic = CriticNetwork(state_dims, action_dims)
        self.target_critic = CriticNetwork(state_dims, action_dims)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic,
                                           weight_decay=config.weight_decay)
        self.noise = OUNoise(config.action_dim)
        self.update_target_networks(tau=1.0)

    def update_target_networks(self, tau: float):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        if explore:
            action += self.noise.noise()
        action = np.clip(action, self.config.action_low[0], self.config.action_high[0])
        return action

    def update(self,
               batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor],
               agents: List['MADDPGAgent']):
        states, actions, rewards, next_states, dones = batch
        agent_states = states[self.agent_id]
        agent_actions = actions[self.agent_id]
        next_agent_states = next_states[self.agent_id]
        target_next_actions = [agent.target_actor(next_states[i]) for i, agent in enumerate(agents)]
        target_critic_input = next_states + target_next_actions
        target_q = rewards.unsqueeze(1) + self.config.gamma * self.target_critic(target_critic_input) * (
            ~dones).unsqueeze(1)
        critic_input = states + actions
        current_q = self.critic(critic_input)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.agent_id == 0 or (self.agent_id % self.config.policy_freq == 0):
            actor_actions = [agent.actor(states[i]) for i, agent in enumerate(agents)]
            actor_critic_input = states + actor_actions
            actor_loss = -self.critic(actor_critic_input).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.update_target_networks(self.config.tau)


class Trainer:
    def __init__(self, env: PowerDemandResponseEnv, agents: List[MADDPGAgent], buffer: MultiAgentReplayBuffer,
                 config: TrainingConfig):
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.config = config
        self.metrics = []

    def train(self):
        for episode in range(self.config.episodes):
            states = self.env.reset()
            total_reward = 0.0
            for step in range(self.config.max_steps_per_episode):
                actions = [agent.get_action(state) for agent, state in zip(self.agents, states)]
                next_states, reward, done, info = self.env.step(actions[0], actions[1])
                self.buffer.add(states, actions, reward, next_states, done)
                if self.buffer.size >= self.config.batch_size:
                    batch = self.buffer.sample(self.config.batch_size)
                    for agent in self.agents:
                        agent.update(batch, self.agents)
                states = next_states
                total_reward += reward
                if done:
                    break
            self.metrics.append(
                {'episode': episode, 'total_reward': total_reward, 'cost': info['cost'], 'carbon': info['carbon'],
                 'sla': info['sla']})
            if (episode + 1) % self.config.save_interval == 0:
                self.save_models(episode)
                self.save_metrics()
            print(
                f'Episode {episode + 1}/{self.config.episodes}, Reward: {total_reward:.2f}, Cost: {info["cost"]:.2f}, Carbon: {info["carbon"]:.2f}, SLA: {info["sla"]:.2f}')

    def save_models(self, episode: int):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(self.config.model_save_path,
