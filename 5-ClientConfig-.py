import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import yaml
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


@dataclass
class ClientConfig:
    client_id: int
    data_path: str
    model_input_dim: int
    model_hidden_dim: int
    model_output_dim: int
    batch_size: int
    epochs_per_round: int
    lr: float
    weight_decay: float
    device: str


@dataclass
class ServerConfig:
    num_clients: int
    aggregation_method: str
    global_model_path: str
    device: str


@dataclass
class TrainingConfig:
    num_rounds: int
    val_interval: int
    save_interval: int
    log_path: str


@dataclass
class MetricsConfig:
    accuracy_weight: float
    loss_weight: float
    privacy_weight: float
    result_save_path: str


@dataclass
class Config:
    client: ClientConfig
    server: ServerConfig
    training: TrainingConfig
    metrics: MetricsConfig


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    client_config = ClientConfig(**data['client'])
    server_config = ServerConfig(**data['server'])
    training_config = TrainingConfig(**data['training'])
    metrics_config = MetricsConfig(**data['metrics'])
    return Config(client=client_config, server=server_config, training=training_config, metrics=metrics_config)


class FederatedModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ClientDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path).values
        self.x = self.data[:, :-1]
        self.y = self.data[:, -1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class FederatedClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = FederatedModel(config.model_input_dim, config.model_hidden_dim, config.model_output_dim).to(
            self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.loss_fn = nn.MSELoss()
        self.dataset = ClientDataset(config.data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)

    def set_weights(self, weights: List[torch.Tensor]) -> None:
        self.model.load_state_dict(weights)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def train_local(self) -> Tuple[float, int]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for epoch in range(self.config.epochs_per_round):
            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.loss_fn(outputs.squeeze(), y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        avg_loss = total_loss / total_samples
        return avg_loss, total_samples


class FederatedServer:
    def __init__(self, config: ServerConfig, client_config: ClientConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_model = FederatedModel(client_config.model_input_dim, client_config.model_hidden_dim,
                                           client_config.model_output_dim).to(self.device)
        self.aggregation_method = config.aggregation_method
        self.global_model_path = config.global_model_path
        os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)

    def broadcast_weights(self) -> Dict[str, torch.Tensor]:
        return self.global_model.state_dict()

    def aggregate_weights(self, client_weights: List[Tuple[Dict[str, torch.Tensor], int]]) -> None:
        if self.aggregation_method == "fedavg":
            total_samples = sum(samples for _, samples in client_weights)
            global_weights = {}
            for key in client_weights[0][0].keys():
                global_weights[key] = torch.zeros_like(client_weights[0][0][key])
                for weights, samples in client_weights:
                    global_weights[key] += weights[key] * samples
                global_weights[key] /= total_samples
            self.global_model.load_state_dict(global_weights)
        elif self.aggregation_method == "fedprox":
            pass
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

    def save_global_model(self, round_num: int) -> None:
        save_path = f"{self.global_model_path}_round_{round_num}.pth"
        torch.save(self.global_model.state_dict(), save_path)


class FederatedTrainer:
    def __init__(self, server: FederatedServer, clients: List[FederatedClient], config: TrainingConfig):
        self.server = server
        self.clients = clients
        self.config = config
        self.metrics = []

    def run_federated_training(self) -> None:
        for round_num in range(self.config.num_rounds):
            global_weights = self.server.broadcast_weights()
            client_updates = []
            round_loss = 0.0
            total_samples = 0
            for client in self.clients:
                client.set_weights(global_weights)
                loss, samples = client.train_local()
                client_updates.append((client.get_weights(), samples))
                round_loss += loss * samples
                total_samples += samples
            avg_round_loss = round_loss / total_samples
            self.server.aggregate_weights(client_updates)
            if (round_num + 1) % self.config.save_interval == 0:
                self.server.save_global_model(round_num + 1)
            self.metrics.append({"round": round_num + 1, "avg_loss": avg_round_loss})
            print(f"Round {round_num + 1}/{self.config.num_rounds}, Avg Loss: {avg_round_loss:.4f}")
        self.save_metrics()

    def save_metrics(self) -> None:
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.config.log_path, index=False)


def main():
    config = load_config()
    clients = []
    for client_id in range(config.server.num_clients):
        client_config = config.client
        client_config.client_id = client_id
        client_config.data_path = f"data/client_{client_id}.csv"
        clients.append(FederatedClient(client_config))
    server = FederatedServer(config.server, config.client)
    trainer = FederatedTrainer(server, clients, config.training)
    trainer.run_federated_training()


if __name__ == "__main__":
    main()
