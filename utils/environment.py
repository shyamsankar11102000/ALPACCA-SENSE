import os
import torch
import random
import numpy as np
import logging
import warnings

def configure_environment(seed: int = 42):
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return device