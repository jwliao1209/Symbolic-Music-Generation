import glob
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.constants import PROJECT_NAME, CHECKPOINT_DIR, CONFIG_FILE
from src.optimizer import get_optimizer
from src.lr_scheduler import get_lr_scheduler
from src.trainer import Trainer
from src.utils import (
    set_random_seeds,
    get_device,
    get_time,
    save_json,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Symbolic Music Generation')

    # dataset setting
    parser.add_argument(
        '--data_path',
        type=str,
        default='data.json',
        help='path of dataset'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for dataloader.',
    )

    # training setting
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['sgd', 'adam', 'adamw'],
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=4e-4,
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='constant',
        choices=['constant', 'step', 'one_cycle', 'cosine_annealing'],
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, get_time())
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_json(vars(args), os.path.join(checkpoint_dir, CONFIG_FILE))

    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        params=Path("path", "to", "saved", "tokenizer.json")
    )
    tokenizer = REMI(config)
    dataset = DatasetMIDI(
        files_paths=list(Path("Pop1K7", "midi_analyzed").glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    train_loader = DataLoader(dataset, batch_size=8, collate_fn=collator)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = get_device()
    optimizer = get_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_lr_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        max_lr=args.lr,
        steps_for_one_epoch=len(train_loader),
        epochs=args.epochs,
    )

    # Prepare logger
    wandb.init(
        project=PROJECT_NAME,
        name=os.path.basename(checkpoint_dir),
        config=vars(args),
    )
    wandb.watch(model, log='all')

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accum_grad_step=1,
        clip_grad_norm=1.0,
        logger=wandb,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(epochs=args.epochs)
