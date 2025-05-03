import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from configs.config import config
from utils.environment import configure_environment
from data.preprocess import clean_text

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.paths.output_dir,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if config.model.use_lora or config.model.use_qlora:
        model = PeftModel.from_pretrained(model, config.paths.output_dir)

    model.eval()
    return model, tokenizer

def generate_prediction(model, tokenizer, prompt: str, max_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.replace(prompt, "").strip().capitalize()

if __name__ == "__main__":
    configure_environment()
    model, tokenizer = load_model_and_tokenizer()

    instruction = "Determine whether the sentiment in this financial news is positive, negative, or neutral:"

    print("📈 Financial Sentiment Inference — type 'exit' to quit")
    while True:
        user_input = input("\n📰 Enter a financial news headline: ").strip()
        if user_input.lower() == "exit":
            break
        cleaned_input = clean_text(user_input)
        full_prompt = f"{instruction} {cleaned_input}"
        prediction = generate_prediction(model, tokenizer, full_prompt)
        print(f"🔍 Predicted Sentiment: {prediction}")