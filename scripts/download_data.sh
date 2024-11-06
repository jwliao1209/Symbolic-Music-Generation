#!/bin/bash

# Download dataset
if [ ! -d "Pop1K7" ]; then
    wget https://zenodo.org/records/13167761/files/Pop1K7.zip
    unzip Pop1K7.zip
    rm Pop1K7.zip
fi
