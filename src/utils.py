import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from easydict import EasyDict


def get_file_paths(folder_path):
    return sorted(
        list(Path(folder_path).glob("*.mid")),
        key=lambda x: int(re.search(r"(\d+)$", x.stem).group(1))
    )


def read_json(path: str) -> Dict[str, Union[str, int, float]]:
    with open(path, "r") as f:
        return json.load(f)


def load_config(path: str) -> EasyDict:
    return EasyDict(read_json(path))


def get_time() -> str:
    return datetime.today().strftime("%m-%d-%H-%M-%S")


def set_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dict_to_device(data: Dict[str, List[float]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}


def get_trucated_idx(generated_tokens, tokenizer, n_target_bar):
    BAR_TOKEN = [v for k, v in tokenizer.vocab.items() if "Bar" in k][0]
    return np.where(
        np.cumsum([np.array(generated_tokens) == BAR_TOKEN]) == n_target_bar
    )[0][0]
