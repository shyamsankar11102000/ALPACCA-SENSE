import optuna
from model.model import load_model, load_tokenizer
from data.loaders import load_datasets
from train.trainer import train_model
from configs.config import config
from utils.environment import configure_environment
import torch
import numpy as np

def objective(trial):
    config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config.training.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    config.model.lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    
    device = configure_environment(config.training.seed)
    
    tokenizer = load_tokenizer()
    
    full_train_dataset = load_datasets(tokenizer, split="train")
    subset_size = int(0.1 * len(full_train_dataset))
    train_dataset = full_train_dataset.shuffle(seed=config.training.seed).select(range(subset_size))
    eval_dataset = load_datasets(tokenizer, split="validation")
    
    model = load_model()
    model.to(device)
    
    config.training.num_train_epochs = 1
    config.training.eval_steps = 50
    config.training.save_steps = 50
    config.training.logging_steps = 50
    
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
    
    eval_results = trainer.evaluate()
    accuracy = eval_results.get("eval_accuracy", 0.0)
    
    del model, trainer
    torch.cuda.empty_cache()
    
    return accuracy

def run_bayesian_optimization(n_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    print("Best hyperparameters:")
    print(f"Learning Rate: {best_trial.params['learning_rate']}")
    print(f"Batch Size: {best_trial.params['batch_size']}")
    print(f"LoRA Rank: {best_trial.params['lora_r']}")
    print(f"Best Accuracy: {best_trial.value}")
    
    config.training.learning_rate = best_trial.params["learning_rate"]
    config.training.batch_size = best_trial.params["batch_size"]
    config.model.lora_r = best_trial.params["lora_r"]
    
    return config

if __name__ == "__main__":
    config = run_bayesian_optimization()
    print("Updated config with best hyperparameters:", config.training.__dict__, config.model.__dict__)