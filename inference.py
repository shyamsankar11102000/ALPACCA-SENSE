# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from configs.config import config

def load_model_and_tokenizer():
    """Loads the base model and applies the LoRA adapter."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)

    # Load base model in half precision if available
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(model, config.model.lora_weights_path)
    model.eval()

    return model, tokenizer

def generate_response(model, tokenizer, instruction: str, input_text: str, max_tokens: int = 64):
    """Generates sentiment classification given a prompt and input."""
    prompt = f"{instruction.strip()} {input_text.strip()}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Optionally post-process to extract just the sentiment label
    response = generated.replace(prompt, "").strip()
    return response

if __name__ == "__main__":
    # Example usage
    instruction = "Determine whether the sentiment in this financial news is positive, negative, or neutral:"
    input_text = "The company’s quarterly profits soared past expectations, boosting investor confidence."

    model, tokenizer = load_model_and_tokenizer()
    prediction = generate_response(model, tokenizer, instruction, input_text)

    print(f"Predicted Sentiment: {prediction}")
