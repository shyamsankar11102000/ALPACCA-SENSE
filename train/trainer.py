import os
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from configs.config import config
from utils.metrics import compute_metrics

def get_training_args(output_dir: str):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        logging_dir=config.paths.log_dir,
        logging_steps=config.training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=config.training.fp16,
        seed=config.training.seed,
        report_to="none",
    )

def train_model(model, tokenizer, train_dataset, eval_dataset):
    args = get_training_args(config.paths.output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(config.paths.output_dir)
    tokenizer.save_pretrained(config.paths.output_dir)
    return trainer