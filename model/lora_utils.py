from peft import LoraConfig, get_peft_model, TaskType
from configs.config import config

def apply_lora(model):
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def apply_qlora(model):
    qlora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()
    return model