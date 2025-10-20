#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p data
cd data

# Download the CSV files
curl -O https://shrec2025.drugdesign.fr/files/train_set.csv
curl -O https://shrec2025.drugdesign.fr/files/test_set.csv

# Download the VTK tar.gz archives
curl -O https://shrec2025.drugdesign.fr/files/train_set.tar.xz
curl -O https://shrec2025.drugdesign.fr/files/test_set.tar.xz

# Extract the archives (into separate folders)
mkdir -p train_set_vtk
mkdir -p test_set_vtk

tar -xf train_set.tar.xz -C train_set_vtk
tar -xf test_set.tar.xz -C test_set_vtk
