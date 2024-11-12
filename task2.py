import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from miditok import REMI, TokenizerConfig
from symusic import Score
from transformers import AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from src.constants import CONFIG_FILE, CKPT_FILE
from src.utils import get_file_paths, get_device, get_trucated_idx, load_config


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Task 2")
    parser.add_argument(
        "--prompt_song_path",
        type=str,
        default="prompt_song",
        help="path of prompt song",
    )
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
        "--output_folder",
        type=str,
        default="results",
    )
    return parser.parse_args()


def truncate_to_nbars(midi_paths, tokenizer, num_bar=8):
    truncated_midi_tokens = []
    for midi_path in midi_paths:
        midi = Score(midi_path)
        tokens = tokenizer.encode(midi)

        bar_count = 0
        truncated_tokens = []
        for token in tokens:
            truncated_tokens.append(token)
            if token == tokenizer["Bar_None"]:
                bar_count += 1

            if bar_count >= num_bar:
                break
        truncated_midi_tokens.append(truncated_tokens)
    return truncated_midi_tokens


if __name__ == "__main__":
    args = parse_arguments()
    save_folder = Path(args.output_folder, Path(args.ckpt_path).name, "task2")
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

    midi_paths = get_file_paths(args.prompt_song_path)
    truncated_midi_tokens = truncate_to_nbars(midi_paths, tokenizer, num_bar=8)

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
        repetition_penalty=1.2,
    )

    for i, data in enumerate(tqdm(truncated_midi_tokens), start=1):
        generated_tokens = data.copy()
        input_ids = torch.tensor(data).unsqueeze(0).long().to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        while generated_tokens.count(BAR_TOKEN) < args.n_target_bar:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )[0].cpu().numpy().tolist()
            new_token = output[input_ids.shape[-1]:]
            generated_tokens.extend(new_token)
            tqdm.write(f"Prompt song {i} new bar: {new_token.count(BAR_TOKEN)}, generated bar: {generated_tokens.count(BAR_TOKEN)}")

            input_ids = torch.tensor(generated_tokens[-500:]).unsqueeze(0).long().to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

        truncated_idx = get_trucated_idx(generated_tokens, tokenizer, args.n_target_bar)
        valid_tokens = [token for token in generated_tokens[:truncated_idx + 1] if token in tokenizer.vocab.values()]
        generated_midi = tokenizer.decode(valid_tokens)
        generated_midi.dump_midi(Path(save_folder, f"song_{i}.mid"))
