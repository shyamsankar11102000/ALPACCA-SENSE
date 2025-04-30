# data/preprocess.py
import re
from transformers import PreTrainedTokenizer

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z0-9.,!?;:\s]", "", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_example(example: dict, tokenizer: PreTrainedTokenizer, max_length: int) -> dict:
    """Preprocess a single example."""
    text = clean_text(example["text"])
    label = example["label"]

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Remove batch dimension and convert to regular dict
    encoded = {k: v.squeeze(0) for k, v in encoded.items()}
    encoded["labels"] = int(label)  # assuming labels are already encoded numerically

    return encoded
