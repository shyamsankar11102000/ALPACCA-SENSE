from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from configs.config import config
from .lora_utils import apply_lora, apply_qlora

def load_tokenizer():
    return AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        cache_dir=config.paths.cache_dir,
    )

    if config.model.use_qlora:
        model = apply_qlora(model)
    elif config.model.use_lora:
        model = apply_lora(model)

    return model