from model.model import load_model, load_tokenizer
from data.loaders import load_datasets
from train.trainer import train_model
from configs.config import config

def main():
    print("🔧 Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("📚 Loading and preprocessing datasets...")
    train_dataset = load_datasets(tokenizer, split="train")
    eval_dataset = load_datasets(tokenizer, split="validation")

    print("🧠 Loading model...")
    model = load_model()

    print("🚀 Starting training...")
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)

    print("✅ Training complete. Model saved to:", config.paths.output_dir)

if __name__ == "__main__":
    main()
