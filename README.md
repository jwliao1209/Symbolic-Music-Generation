# Symbolic Music Generation

This repository contains the implementation for Homework 3 of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this [slides](https://docs.google.com/presentation/d/1f27a5Ok4PWTeoof0pUThKlr9Kegm_fX5AYOJuFv4s_w/edit?usp=sharing).


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 deepmir_hw3
source deepmir_hw3/bin/activate
pip install -r requirements.txt
```


## Data and Checkpoint Download

### Dataset
To download the dataset, run the following script:
```
bash scripts/download_data.sh
```

### Checkpoint
To download the pre-trained model checkpoints, use the command:
```
bash scripts/download_ckpt.sh
```


### Sound Font
To download the sound font, use the command:
```
bash scripts/download_soundfont.sh
```


## Training
To train the model, run the command:
```
python train.py
```


## Inference

### Task 1
To generate the symbolic music, run the command:
```
python task1.py \
    --ckpt_path checkpoints/<checkpoint name> \
    --output_folder <Folder path for saving the generated file>
```

### Task 2
To continue the symbolic music, run the command:
```
python task2.py \
    --ckpt_path checkpoints/<checkpoint name> \
    --prompt_song_path <Folder path for prompt song> \
    --output_folder <Folder path for saving the generated file>
```


## Evaluation
To evaluate the generated results, run the command:
```
python eval.py \
    --eval_folder <Results folder to evaluation> \
    --output_path <Path for output score.csv>
```


## Converting MIDI Files to WAV Format
To convert `.mid` file to `.wav` file, run the command:
```
python convert_mid_to_wav.py \
    --input_folder <Path for input folder> \
    --output_folder <Path for output folder>
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{liao2024_source_separation,
    title  = {Symbolic Music Generation},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Symbolic-Music-Generation},
    year   = {2024}
}
```
