import json
import random
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import torch


def read_json(path: str) -> Dict[str, Union[str, int, float]]:
    with open(path, 'r') as f:
        return json.load(f)


def get_time() -> str:
    return datetime.today().strftime('%m-%d-%H-%M-%S')


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
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dict_to_device(data: Dict[str, List[float]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}
