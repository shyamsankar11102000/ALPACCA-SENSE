import re
import random
from transformers import PreTrainedTokenizer
PROMPTS = [
    "Classify the sentiment of the following headline as positive, negative, or neutral:",
    "Determine whether the sentiment in this financial news is positive, negative, or neutral:",
    "Does the following headline express a positive, negative, or neutral sentiment?",
    "Identify if the sentiment of the headline below is positive, negative, or neutral:",
    "Analyze the sentiment (positive, negative, or neutral) conveyed in this financial headline:",
    "Based on the content, is the sentiment of the following headline positive, negative, or neutral?",
    "Evaluate the tone of the following market news as positive, negative, or neutral:",
    "Label the sentiment of this business headline as either positive, negative, or neutral:",
    "Decide if the sentiment behind this economic news is positive, negative, or neutral:",
    "Classify the overall sentiment in the headline below as positive, negative, or neutral:",
    "Is the sentiment of the following financial update best described as positive, negative, or neutral?",
    "Assign one of the following sentiment labels to this news headline: positive, negative, or neutral:",
    "Indicate whether the market sentiment in the headline is positive, negative, or neutral:",
    "What sentiment — positive, negative, or neutral — is most reflected in the following headline?",
    "Given this financial headline, categorize the sentiment as positive, negative, or neutral:"
]

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9.,!?;:\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_example(example: dict, tokenizer: PreTrainedTokenizer, max_length: int) -> dict:
    """Preprocess a single example with instruction prompt."""
    prompt = random.choice(PROMPTS)
    instruction_input = f"{prompt} {clean_text(example['input'])}"  # 'input' key holds the headline
    label = example["output"]  # assumed to be already string label ("Positive", etc.)

    encoded = tokenizer(
        instruction_input,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encoded = {k: v.squeeze(0) for k, v in encoded.items()}
    
    # You may use a mapping like: {"positive": 0, "neutral": 1, "negative": 2}
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    encoded["labels"] = label_map[label.lower()]

    return encoded
