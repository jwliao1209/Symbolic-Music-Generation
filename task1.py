import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from miditok import REMI, TokenizerConfig
from transformers import AutoModelForCausalLM, GenerationConfig
from tqdm import trange, tqdm

from src.constants import CKPT_FILE, CONFIG_FILE
from src.utils import (
    set_random_seeds,
    get_device,
    get_trucated_idx,
    load_config,
    generate_tokens,
    filter_invalid_tokens,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Task 1")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/11-07-23-27-52",
        help="path of checkpoint",
    )
    parser.add_argument(
        "--n_target_bar",
        type=int,
        default=32,
        help="number of target bars",
    )
    parser.add_argument(
        "--n_generated_midi",
        type=int,
        default=20,
        help="number of generated midi",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    save_folder = Path(args.output_folder, Path(args.ckpt_path).name, "task1")
    os.makedirs(save_folder, exist_ok=True)
    ckpt_config = load_config(Path(args.ckpt_path, CONFIG_FILE))
    tokenizer_config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        use_tempos=True,
        params=Path(args.ckpt_path, "tokenizer.json")
    )
    tokenizer = REMI(tokenizer_config)
    BAR_TOKEN = [v for k, v in tokenizer.vocab.items() if "Bar" in k][0]

    model = AutoModelForCausalLM.from_pretrained(ckpt_config.model_name)
    model.load_state_dict(
        torch.load(Path(args.ckpt_path, CKPT_FILE), weights_only=True)["model"]
    )
    device = get_device()
    model.to(device)
    model.eval()

    generation_config = GenerationConfig(
        max_length=1024,
        do_sample=True,
        temperature=1.2,
        top_k=5,
        pad_token_id=model.config.eos_token_id,
        repetition_penalty=1.5,
    )

    for i in trange(1, args.n_generated_midi + 1):
        generated_tokens = generate_tokens([1], model, device, args.n_target_bar, BAR_TOKEN, generation_config)
        truncated_idx = get_trucated_idx(generated_tokens, tokenizer, args.n_target_bar)
        valid_tokens = filter_invalid_tokens(generated_tokens[:truncated_idx + 1], tokenizer)
        generated_midi = tokenizer.decode(valid_tokens)
        generated_midi.dump_midi(Path(save_folder, f"output_{i}.mid"))
