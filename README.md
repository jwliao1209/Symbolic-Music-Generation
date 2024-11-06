# Symbolic Music Generation

This repository contains the implementation for Homework 3 of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this [slides]().


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


## Training
To train the model, run the command:
```
bash scripts/train.sh
```


## Inference
To inference the model, run the command:
```
bash scripts/inference.sh
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
