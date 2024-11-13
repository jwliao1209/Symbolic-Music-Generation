#!/bin/bash

# Download checkpoints
if [ ! -d "checkpoints" ]; then
    gdown 19JSJNpmUMwYgd6xHr8n4nV80YQKiIzcT -O checkpoints.zip
    unzip -n checkpoints.zip
    rm checkpoints.zip
fi
