# data/loaders.py
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizer
from .preprocess import preprocess_example
from configs.config import config

def load_datasets(tokenizer: PreTrainedTokenizer, split="train") -> DatasetDict:
    datasets = []

    if "FiQA" in config.dataset.dataset_names:
        fiqa = load_dataset("financial_phrasebank", split=split)
        datasets.append(fiqa)

    if "FinancialPhraseBank" in config.dataset.dataset_names:
        fpb = load_dataset("financial_phrasebank", "sentences_allagree", split=split)
        datasets.append(fpb)

    if not datasets:
        raise ValueError("No valid datasets found in config.")

    # Merge datasets
    full_dataset = concatenate_datasets(datasets)

    # Map preprocessing function
    processed_dataset = full_dataset.map(
        lambda x: preprocess_example(x, tokenizer, config.dataset.max_seq_length),
        remove_columns=full_dataset.column_names,
    )

    return processed_dataset
