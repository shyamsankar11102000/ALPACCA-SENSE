from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from configs.config import config
from model.lora_utils import apply_lora, apply_qlora
from data.loaders import load_datasets
from train.trainer import train_model
from utils.environment import configure_environment
import logging

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        cache_dir=config.paths.cache_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    if config.model.use_qlora:
        model = apply_qlora(model)
    elif config.model.use_lora:
        model = apply_lora(model)

    return model

def train_and_save_model():
    device = configure_environment(config.training.seed)
    
    logging.info("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    logging.info("Loading and preprocessing datasets...")
    train_dataset = load_datasets(tokenizer, split="train")
    eval_dataset = load_datasets(tokenizer, split="validation")
    
    logging.info("Loading model...")
    model = load_model()
    model.to(device)
    
    logging.info("Starting training...")
    config.training.save_steps = 10
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
    
    logging.info("Evaluating model on validation dataset...")
    eval_results = trainer.evaluate()
    f1_score = eval_results.get("eval_f1", 0.0)
    accuracy = eval_results.get("eval_accuracy", 0.0)
    logging.info(f"Validation F1 Score: {f1_score:.4f}")
    logging.info(f"Validation Accuracy: {accuracy:.4f}")
    
    logging.info("Saving model and tokenizer...")
    trainer.save_model(config.paths.output_dir)
    tokenizer.save_pretrained(config.paths.output_dir)
    
    logging.info(f"Training complete. Model saved to: {config.paths.output_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_and_save_model()