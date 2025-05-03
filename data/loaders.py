from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizer
from .preprocess import preprocess_example
from configs.config import config

def load_datasets(tokenizer: PreTrainedTokenizer, split="train") -> DatasetDict:
    datasets = []

    if split == "train":
        if "TwitterFinancialNews" in config.dataset.dataset_names:
            tfn = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
            datasets.append(tfn)
        if "FiQA" in config.dataset.dataset_names:
            fiqa = load_dataset("financial_phrasebank", split="train")
            datasets.append(fiqa)
    elif split == "validation":
        if "TwitterFinancialNews" in config.dataset.dataset_names:
            tfn_val = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
            datasets.append(tfn_val)
    elif split == "test":
        if "FinancialPhraseBank" in config.dataset.dataset_names:
            fpb = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
            datasets.append(fpb)

    if not datasets:
        raise ValueError(f"No valid datasets found for {split} split in config.")

    full_dataset = concatenate_datasets(datasets)

    processed_dataset = full_dataset.map(
        lambda x: preprocess_example(x, tokenizer, config.dataset.max_seq_length),
        remove_columns=full_dataset.column_names,
    )

    return processed_dataset