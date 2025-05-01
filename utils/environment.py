# utils/environment.py

import os
import torch
import random
import numpy as np
import logging
import warnings

def configure_environment(seed: int = 42):
    """Set up environment for reproducible and efficient training/inference."""
    
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable faster transformer operations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Print available device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Optional: reduce tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return device
