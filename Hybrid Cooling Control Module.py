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
    rack_volume: float
    rack_heat_capacity: float
    ambient_temp: float
    max_tec_current: float
    min_tec_current: float
    max_water_flow: float
    min_water_flow: float
    max_water_inlet_temp: float
    min_water_inlet_temp: float
    tec_seebeck_coeff: float
    tec_resistance: float
    tec_thermal_conductivity: float
    water_density: float
    water_specific_heat: float
    water_heat_transfer_coeff: float
    heat_dissipation_area: float
    it_load_base: float
    it_load_variance: float
    target_rack_temp: float


@dataclass
class UpperAgentConfig:
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
class LowerAgentConfig:
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
    upper_episodes: int
    lower_episodes_per_upper: int
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
    energy_cost_weight: float
    temp_deviation_weight: float
    cop_weight: float
    result_save_path: str


@dataclass
class Config:
    env: EnvConfig
    upper_agent: UpperAgentConfig
    lower_agent: LowerAgentConfig
    training: TrainingConfig
    metrics: MetricsConfig


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    env_config = EnvConfig(**data['env'])
    upper_agent_config = UpperAgentConfig(**data['upper_agent'])
    lower_agent_config = LowerAgentConfig(**data['lower_agent'])
    training_config = TrainingConfig(**data['training'])
    metrics_config = MetricsConfig(**data['metrics'])
    return Config(env=env_config, upper_agent=upper_agent_config, lower_agent=lower_agent_config,
                  training=training_config, metrics=metrics_config)


class HybridCoolingEnv:
    def __init__(self, config: EnvConfig):
        self.rack_volume = config.rack_volume
        self.rack_heat_capacity = config.rack_heat_capacity
        self.ambient_temp = config.ambient_temp
        self.max_tec_current = config.max_tec_current
        self.min_tec_current = config.min_tec_current
        self.max_water_flow = config.max_water_flow
        self.min_water_flow = config.min_water_flow
        self.max_water_inlet_temp = config.max_water_inlet_temp
        self.min_water_inlet_temp = config.min_water_inlet_temp
        self.tec_seebeck = config.tec_seebeck_coeff
        self.tec_resistance = config.tec_resistance
        self.tec_thermal_conductivity = config.tec_thermal_conductivity
        self.water_density = config.water_density
        self.water_specific_heat = config.water_specific_heat
        self.water_htc = config.water_heat_transfer_coeff
        self.heat_area = config.heat_dissipation_area
        self.it_load_base = config.it_load_base
        self.it_load_var = config.it_load_variance
        self.target_temp = config.target_rack_temp
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.rack_temp = np.random.uniform(28.0, 35.0)
        self.tec_current = np.random.uniform(self.min_tec_current, self.max_tec_current / 2)
        self.water_flow = np.random.uniform(self.min_water_flow, self.max_water_flow / 2)
        self.water_inlet_temp = np.random.uniform(self.min_water_inlet_temp, self.max_water_inlet_temp / 2)
        self.tec_cold_temp = self.rack_temp - np.random.uniform(1.0, 3.0)
        self.tec_hot_temp = self.tec_cold_temp + np.random.uniform(5.0, 10.0)
        self.water_outlet_temp = self.water_inlet_temp + np.random.uniform(0.5, 2.0)
        self.it_load = self.it_load_base + np.random.normal(0, self.it_load_var)
        self.energy_tec = 0.0
        self.energy_water = 0.0
        self.cop_tec = 0.0
        self.step_count = 0
        return self.get_state()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        upper_state = np.array(
            [self.rack_temp, self.it_load, self.water_flow, self.water_inlet_temp, self.water_outlet_temp,
             self.cop_tec])
        lower_state = np.array(
            [self.rack_temp, self.tec_current, self.tec_cold_temp, self.tec_hot_temp, self.water_flow,
             self.water_inlet_temp, self.it_load, self.cop_tec])
        return upper_state, lower_state

    def compute_tec_physics(self) -> Tuple[float, float, float, float]:
        delta_T = self.tec_hot_temp - self.tec_cold_temp
        q_c = self.tec_seebeck * self.tec_current * self.tec_cold_temp - 0.5 * self.tec_resistance * (
                    self.tec_current ** 2) - self.tec_thermal_conductivity * delta_T
        q_h = self.tec_seebeck * self.tec_current * self.tec_hot_temp + 0.5 * self.tec_resistance * (
                    self.tec_current ** 2) - self.tec_thermal_conductivity * delta_T
        p_tec = self.tec_resistance * (self.tec_current ** 2)
        cop = q_c / p_tec if p_tec != 0 else 0.1
        return q_c, q_h, p_tec, cop

    def compute_water_physics(self) -> Tuple[float, float, float, float]:
        q_water = self.water_flow * self.water_density * self.water_specific_heat * (
                    self.water_outlet_temp - self.water_inlet_temp)
        q_conv = self.water_htc * self.heat_area * (self.rack_temp - self.water_inlet_temp)
        p_water = 0.001 * (self.water_flow ** 3) * (self.water_outlet_temp - self.water_inlet_temp)
        new_outlet_temp = self.water_inlet_temp + (q_conv) / (
                    self.water_flow * self.water_density * self.water_specific_heat) if self.water_flow != 0 else self.water_inlet_temp
        new_outlet_temp = np.clip(new_outlet_temp, self.water_inlet_temp, self.max_water_inlet_temp + 5.0)
        return q_water, q_conv, p_water, new_outlet_temp

    def compute_heat_generation(self) -> float:
        heat = self.it_load + 0.1 * (self.rack_temp - self.ambient_temp) * self.rack_volume
        return max(1000.0, heat)

    def step_upper(self, upper_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        flow_delta = upper_action[0]
        inlet_temp_delta = upper_action[1]
        self.water_flow = np.clip(self.water_flow + flow_delta, self.min_water_flow, self.max_water_flow)
        self.water_inlet_temp = np.clip(self.water_inlet_temp + inlet_temp_delta, self.min_water_inlet_temp,
                                        self.max_water_inlet_temp)
        heat_gen = self.compute_heat_generation()
        q_tec, q_h_tec, p_tec, cop_tec = self.compute_tec_physics()
        q_water, q_conv_water, p_water, new_outlet = self.compute_water_physics()
        total_cooling = q_tec + q_conv_water
        temp_change = (heat_gen - total_cooling) / (self.rack_heat_capacity * self.rack_volume)
        self.rack_temp += temp_change
        self.rack_temp = np.clip(self.rack_temp, 25.0, 40.0)
        self.water_outlet_temp = new_outlet
        self.tec_hot_temp = self.water_outlet_temp + np.random.uniform(1.0, 2.0)
        self.tec_cold_temp = self.rack_temp - (q_tec) / (self.tec_thermal_conductivity * self.heat_area) if (
                                                                                                                        self.tec_thermal_conductivity * self.heat_area) != 0 else self.rack_temp - 1.0
        self.tec_cold_temp = np.clip(self.tec_cold_temp, 20.0, self.rack_temp - 0.5)
        self.energy_tec += p_tec
        self.energy_water += p_water
        self.cop_tec = cop_tec
        temp_dev = abs(self.rack_temp - self.target_temp)
        reward = - (0.4 * p_tec + 0.3 * p_water + 0.2 * temp_dev + 0.1 * (1.0 - cop_tec))
        done = self.step_count >= 300 or self.rack_temp >= 40.0 or self.rack_temp <= 25.0
        self.step_count += 1
        upper_state, _ = self.get_state()
        info = {'heat_gen': heat_gen, 'total_cooling': total_cooling, 'cop_tec': cop_tec, 'energy_tec': self.energy_tec,
                'energy_water': self.energy_water}
        return upper_state, reward, done, info

    def step_lower(self, lower_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        current_delta = lower_action[0]
        self.tec_current = np.clip(self.tec_current + current_delta, self.min_tec_current,
                                   self.max_tec_current) < / think_never_used_51bce0c785ca2f68081bfa7d91973934 > < / think_never_used_51bce0c785ca2f68081bfa7d91973934 >


```python
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
    rack_volume: float
    rack_heat_capacity: float
    ambient_temp: float
    max_tec_current: float
    min_tec_current: float
    max_water_flow: float
    min_water_flow: float
    max_water_inlet_temp: float
    min_water_inlet_temp: float
    tec_seebeck_coeff: float
    tec_resistance: float
    tec_thermal_conductivity: float
    water_density: float
    water_specific_heat: float
    water_heat_transfer_coeff: float
    heat_dissipation_area: float
    it_load_base: float
    it_load_variance: float
    target_rack_temp: float


@dataclass
class UpperAgentConfig:
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
class LowerAgentConfig:
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
    upper_episodes: int
    lower_episodes_per_upper: int
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
    energy_cost_weight: float
    temp_deviation_weight: float
    cop_weight: float
    result_save_path: str


@dataclass
class Config:
    env: EnvConfig
    upper_agent: UpperAgentConfig
    lower_agent: LowerAgentConfig
    training: TrainingConfig
    metrics: MetricsConfig


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    env_config = EnvConfig(**data['env'])
    upper_agent_config = UpperAgentConfig(**data['upper_agent'])
    lower_agent_config = LowerAgentConfig(**data['lower_agent'])
    training_config = TrainingConfig(**data['training'])
    metrics_config = MetricsConfig(**data['metrics'])
    return Config(env=env_config, upper_agent=upper_agent_config, lower_agent=lower_agent_config,
                  training=training_config, metrics=metrics_config)


class HybridCoolingEnv:
    def __init__(self, config: EnvConfig):
        self.rack_volume = config.rack_volume
        self.rack_heat_capacity = config.rack_heat_capacity
        self.ambient_temp = config.ambient_temp
        self.max_tec_current = config.max_tec_current
        self.min_tec_current = config.min_tec_current
        self.max_water_flow = config.max_water_flow
        self.min_water_flow = config.min_water_flow
        self.max_water_inlet_temp = config.max_water_inlet_temp
        self.min_water_inlet_temp = config.min_water_inlet_temp
        self.tec_seebeck = config.tec_seebeck_coeff
        self.tec_resistance = config.tec_resistance
        self.tec_thermal_conductivity = config.tec_thermal_conductivity
        self.water_density = config.water_density
        self.water_specific_heat = config.water_specific_heat
        self.water_htc = config.water_heat_transfer_coeff
        self.heat_area = config.heat_dissipation_area
        self.it_load_base = config.it_load_base
        self.it_load_var = config.it_load_variance
        self.target_temp = config.target_rack_temp
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.rack_temp = np.random.uniform(28.0, 35.0)
        self.tec_current = np.random.uniform(self.min_tec_current, self.max_tec_current / 2)
        self.water_flow = np.random.uniform(self.min_water_flow, self.max_water_flow / 2)
        self.water_inlet_temp = np.random.uniform(self.min_water_inlet_temp, self.max_water_inlet_temp / 2)
        self.tec_cold_temp = self.rack_temp - np.random.uniform(1.0, 3.0)
        self.tec_hot_temp = self.tec_cold_temp + np.random.uniform(5.0, 10.0)
        self.water_outlet_temp = self.water_inlet_temp + np.random.uniform(0.5, 2.0)
        self.it_load = self.it_load_base + np.random.normal(0, self.it_load_var)
        self.energy_tec = 0.0
        self.energy_water = 0.0
        self.cop_tec = 0.0
        self.step_count = 0
        return self.get_state()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        upper_state = np.array(
            [self.rack_temp, self.it_load, self.water_flow, self.water_inlet_temp, self.water_outlet_temp,
             self.cop_tec])
        lower_state = np.array(
            [self.rack_temp, self.tec_current, self.tec_cold_temp, self.tec_hot_temp, self.water_flow,
             self.water_inlet_temp, self.it_load, self.cop_tec])
        return upper_state, lower_state

    def compute_tec_physics(self) -> Tuple[float, float, float, float]:
        delta_T = self.tec_hot_temp - self.tec_cold_temp
        q_c = self.tec_seebeck * self.tec_current
