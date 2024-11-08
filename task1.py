import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from miditok import REMI, TokenizerConfig
from transformers import AutoModelForCausalLM, GenerationConfig
from tqdm import trange

from src.utils import set_random_seeds, get_device, get_trucated_idx


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Task 1")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/11-07-23-27-52/checkpoint.pt",
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
    return parser.parse_args()


if __name__ == '__main__':
    set_random_seeds()
    args = parse_arguments()
    os.makedirs("results", exist_ok=True)

    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        use_tempos=True,
        params=Path("path", "to", "save", "tokenizer.json")
    )
    tokenizer = REMI(config)
    BAR_TOKEN = [v for k, v in tokenizer.vocab.items() if "Bar" in k][0]

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.load_state_dict(
        torch.load(args.ckpt_path, weights_only=True)["model"]
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

    for i in trange(args.n_generated_midi):
        generated_tokens = [1]
        input_ids = torch.tensor([1]).unsqueeze(0).long().to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        while generated_tokens.count(BAR_TOKEN) < args.n_target_bar:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )[0].cpu().numpy().tolist()
            new_token = output[input_ids.shape[-1]:]
            generated_tokens.extend(new_token)

            input_ids = torch.tensor(generated_tokens[-100:]).unsqueeze(0).long().to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

        truncated_idx = get_trucated_idx(generated_tokens, tokenizer, args.n_target_bar)
        valid_tokens = [token for token in generated_tokens[:truncated_idx + 1] if token in tokenizer.vocab.values()]
        generated_midi = tokenizer.decode(valid_tokens)
        generated_midi.dump_midi(f'results/output_{i}.mid')
