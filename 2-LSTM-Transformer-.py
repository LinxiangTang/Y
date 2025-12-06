# 数据配置
data:
  train_path: "../data/train_load.csv"       # 训练数据路径
  val_path: "../data/val_load.csv"           # 验证数据路径
  test_path: "../data/test_load.csv"         # 测试数据路径
  seq_len: 30                                # 输入序列长度（论文2.1节：30分钟窗口）
  pred_len: 1                                # 预测序列长度（预测未来1步）
  target_col: "cpu_utilization"              # 目标列（要预测的负载指标）
  time_col: "timestamp"                      # 时间戳列
  features: ["cpu_utilization", "rack_temp", "task_count", "power_consumption"]  # 输入特征列
  missing_value_strategy: "interpolate"      # 缺失值处理策略：interpolate/mean/median
  scaling_strategy: "standard"               # 特征缩放策略：standard/minmax

# 模型配置
model:
  lstm_hidden_dim: 128                       # LSTM隐藏层维度
  lstm_num_layers: 2                         # LSTM层数
  lstm_dropout: 0.2                          # LSTM dropout率
  lstm_bidirectional: False                  # 是否双向LSTM
  transformer_d_model: 128                   # Transformer输入维度（需与LSTM输出匹配）
  transformer_nhead: 4                       # 多头注意力头数
  transformer_num_layers: 2                  # Transformer编码器层数
  transformer_dim_feedforward: 512           # Transformer前馈层维度
  transformer_dropout: 0.2                   # Transformer dropout率
  output_dim: 1                              # 输出维度（单变量预测）

# 训练配置
train:
  batch_size: 64                             # 批次大小
  epochs: 100                                # 最大训练轮数
  learning_rate: 0.001                       # 初始学习率
  weight_decay: 0.0001                       # L2正则化
  early_stop_patience: 10                    # 早停耐心值
  save_best_model: True                      # 是否保存最佳模型
  model_save_path: "../models/load_prediction_best.pth"  # 最佳模型保存路径
  log_path: "../logs/load_prediction.log"    # 日志保存路径

# 推理配置
infer:
  model_load_path: "../models/load_prediction_best.pth"  # 推理模型路径
  result_save_path: "../results/prediction_results.csv"  # 预测结果保存路径
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    seq_len: int
    pred_len: int
    target_col: str
    time_col: str
    features: List[str]
    missing_value_strategy: str
    scaling_strategy: str


@dataclass
class ModelConfig:
    lstm_hidden_dim: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_bidirectional: bool
    transformer_d_model: int
    transformer_nhead: int
    transformer_num_layers: int
    transformer_dim_feedforward: int
    transformer_dropout: float
    output_dim: int


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stop_patience: int
    save_best_model: bool
    model_save_path: str
    log_path: str


@dataclass
class InferConfig:
    model_load_path: str
    result_save_path: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    infer: InferConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 解析各子配置
    data_config = DataConfig(**config_dict["data"])
    model_config = ModelConfig(**config_dict["model"])
    train_config = TrainConfig(**config_dict["train"])
    infer_config = InferConfig(**config_dict["infer"])

    return Config(data=data_config, model=model_config, train=train_config, infer=infer_config)


# 测试配置加载
if __name__ == "__main__":
    config = load_config()
    print(config.data.seq_len)  # 输出30
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    seq_len: int
    pred_len: int
    target_col: str
    time_col: str
    features: List[str]
    missing_value_strategy: str
    scaling_strategy: str


@dataclass
class ModelConfig:
    lstm_hidden_dim: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_bidirectional: bool
    transformer_d_model: int
    transformer_nhead: int
    transformer_num_layers: int
    transformer_dim_feedforward: int
    transformer_dropout: float
    output_dim: int


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stop_patience: int
    save_best_model: bool
    model_save_path: str
    log_path: str


@dataclass
class InferConfig:
    model_load_path: str
    result_save_path: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    infer: InferConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 解析各子配置
    data_config = DataConfig(**config_dict["data"])
    model_config = ModelConfig(**config_dict["model"])
    train_config = TrainConfig(**config_dict["train"])
    infer_config = InferConfig(**config_dict["infer"])

    return Config(data=data_config, model=model_config, train=train_config, infer=infer_config)


# 测试配置加载
if __name__ == "__main__":
    config = load_config()
    print(config.data.seq_len)  # 输出30
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.feature_names: List[str] = []
        self.target_name: str = config.target_col

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            df = pd.read_csv(file_path, parse_dates=[self.config.time_col], index_col=self.config.time_col)
            logger.info(f"Loaded data from {file_path}, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            logger.info("No missing values found.")
            return df

        logger.info(f"Handling missing values for columns: {missing_cols}")
        if self.config.missing_value_strategy == "interpolate":
            df = df.interpolate(method="linear", limit_direction="both")
        elif self.config.missing_value_strategy == "mean":
            df = df.fillna(df.mean())
        elif self.config.missing_value_strategy == "median":
            df = df.fillna(df.median())
        else:
            raise ValueError(f"Unsupported missing value strategy: {self.config.missing_value_strategy}")

        logger.info(f"Missing values handled. Remaining missing values: {df.isnull().sum().sum()}")
        return df

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征（年/月/日/时/分/星期/是否节假日）"""
        df = df.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["weekday"] = df.index.weekday  # 0=周一，6=周日
        df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # 模拟节假日（示例：2023年10月1日为节假日）
        holidays = pd.to_datetime(["2023-10-01", "2023-12-25", "2024-01-01"])
        df["is_holiday"] = df.index.isin(holidays).astype(int)

        self.feature_names.extend(["year", "month", "day", "hour", "minute", "weekday", "is_weekend", "is_holiday"])
        logger.info(f"Extracted time features: {self.feature_names[-8:]}")
        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """特征缩放"""
        feature_cols = self.config.features + self.feature_names
        if fit:
            if self.config.scaling_strategy == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaling_strategy == "minmax":
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                raise ValueError(f"Unsupported scaling strategy: {self.config.scaling_strategy}")

            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            logger.info(f"Scaler fitted on training data. Scaling strategy: {self.config.scaling_strategy}")
        else:
            if not self.scaler:
                raise RuntimeError("Scaler not fitted! Call scale_features with fit=True first.")
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            logger.info(f"Scaled data using pre-fitted scaler.")

        return df

    def generate_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """生成序列数据（滑动窗口）"""
        feature_cols = self.config.features + self.feature_names
        target_col = self.config.target_col

        X, y = [], []
        seq_len = self.config.seq_len
        pred_len = self.config.pred_len

        for i in range(len(df) - seq_len - pred_len + 1):
            # 输入序列：[i, i+seq_len)
            x_seq = df[feature_cols].iloc[i:i + seq_len].values
            # 目标序列：[i+seq_len, i+seq_len+pred_len)
            y_seq = df[target_col].iloc[i + seq_len:i + seq_len + pred_len].values

            X.append(x_seq)
            y.append(y_seq)

        X = np.array(X)  # shape: (num_samples, seq_len, num_features)
        y = np.array(y)  # shape: (num_samples, pred_len)

        logger.info(f"Generated sequences: X shape={X.shape}, y shape={y.shape}")
        return X, y

    def preprocess_train_data(self, train_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """训练数据完整预处理流程"""
        logger.info("Starting train data preprocessing...")
        df = self.load_data(train_path)
        df = self.handle_missing_values(df)
        df = self.extract_time_features(df)
        df = self.scale_features(df, fit=True)
        X_train, y_train = self.generate_sequences(df)
        logger.info("Train data preprocessing completed.")
        return X_train, y_train

    def preprocess_val_test_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """验证/测试数据完整预处理流程（复用训练集scaler）"""
        logger.info(f"Starting val/test data preprocessing for {file_path}...")
        df = self.load_data(file_path)
        df = self.handle_missing_values(df)
        df = self.extract_time_features(df)
        df = self.scale_features(df, fit=False)
        X_data, y_data = self.generate_sequences(df)
        logger.info(f"Val/test data preprocessing completed.")
        return X_data, y_data

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """目标值逆缩放（还原真实值）"""
        if not self.scaler:
            raise RuntimeError("Scaler not fitted!")

        # 构造与scaler输入维度匹配的数组（仅目标列位置有效）
        dummy = np.zeros((y_scaled.shape[0], len(self.scaler.feature_names_in_)))
        dummy[:, self.scaler.feature_names_in_.tolist().index(self.target_name)] = y_scaled.flatten()
        y_inv = self.scaler.inverse_transform(dummy)[:, self.scaler.feature_names_in_.tolist().index(self.target_name)]
        return y_inv.reshape(y_scaled.shape)


# 测试数据预处理
if __name__ == "__main__":
    from config import load_config

    config = load_config()
    processor = DataProcessor(config.data)

    # 模拟生成测试数据
    test_df = pd.DataFrame(
        data={
            "cpu_utilization": np.random.rand(1000) * 100,
            "rack_temp": np.random.rand(1000) * 10 + 25,
            "task_count": np.random.randint(0, 100, 1000),
            "power_consumption": np.random.rand(1000) * 500
        },
        index=pd.date_range(start="2023-01-01", periods=1000, freq="5min")
    )
    test_df.to_csv("../data/test_load.csv")

    # 测试预处理流程
    X_test, y_test = processor.preprocess_val_test_data("../data/test_load.csv")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class LoadPredictionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: 输入序列，shape=(num_samples, seq_len, num_features)
            y: 目标序列，shape=(num_samples, pred_len)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# 测试Dataset
if __name__ == "__main__":
    X = np.random.rand(100, 30, 12)  # 100样本，30序列长度，12特征
    y = np.random.rand(100, 1)  # 100样本，1预测长度
    dataset = LoadPredictionDataset(X, y)
    print(f"Dataset length: {len(dataset)}")
    x_sample, y_sample = dataset[0]
    print(f"Sample X shape: {x_sample.shape}, Sample y shape: {y_sample.shape}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config import ModelConfig


class LSTMTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # -------------------------- LSTM层 --------------------------
        self.lstm_input_dim = len(config.features) + 8  # 原始特征+8个时间特征（来自data_processor）
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional,
            batch_first=True
        )

        # LSTM输出维度调整（双向则乘2）
        self.lstm_output_dim = config.lstm_hidden_dim * (2 if config.lstm_bidirectional else 1)

        # -------------------------- Transformer层 --------------------------
        # 输入维度匹配：LSTM输出 → Transformer输入
        self.transformer_input_proj = nn.Linear(self.lstm_output_dim, config.transformer_d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
                batch_first=True,
                norm_first=True  # 先归一化再计算
            ),
            num_layers=config.transformer_num_layers,
            norm=nn.LayerNorm(config.transformer_d_model)
        )


