#!/bin/bash

CKPT_NAME=${1:-"11-12-23-45-22"}
python task2.py \
    --ckpt_path checkpoints/$CKPT_NAME

python convert_mid_to_wav.py \
    --input_folder results/$CKPT_NAME/task2 \
    --output_folder results/$CKPT_NAME/task2_wav
