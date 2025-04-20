#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p data
cd data

# Download the CSV files
curl -O https://shrec2025.drugdesign.fr/files/train_set.csv
curl -O https://shrec2025.drugdesign.fr/files/test_set.csv

# # Download the VTK tar.gz archives
# curl -O https://shrec2025.drugdesign.fr/files/train_set_vtk.tar.gz
# curl -O https://shrec2025.drugdesign.fr/files/test_set_vtk.tar.gz

# # Extract the archives (into separate folders)
# mkdir -p train_set_vtk
# mkdir -p test_set_vtk

# tar -xzf train_set_vtk.tar.gz -C train_set_vtk
# tar -xzf test_set_vtk.tar.gz -C test_set_vtk
