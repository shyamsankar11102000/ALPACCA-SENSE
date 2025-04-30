# configs/config.py

import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3-3B"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = ("q_proj", "v_proj")  # Adjust for transformer arch

@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True  # Enable mixed precision

@dataclass
class DatasetConfig:
    dataset_names: list = ("FiQA", "TwitterTrain")
    max_seq_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    label_column: str = "label"
    text_column: str = "text"

@dataclass
class Paths:
    base_dir: str = os.getcwd()
    output_dir: str = os.path.join(base_dir, "outputs")
    cache_dir: str = os.path.join(base_dir, "cache")
    log_dir: str = os.path.join(base_dir, "logs")
    pretrained_model_dir: str = os.path.join(base_dir, "pretrained")
    dataset_dir: str = os.path.join(base_dir, "datasets")

# Aggregate config for convenient import
@dataclass
class Config:
    model = ModelConfig()
    training = TrainingConfig()
    dataset = DatasetConfig()
    paths = Paths()

config = Config()
