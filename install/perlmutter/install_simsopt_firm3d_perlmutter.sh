#!/bin/bash

# load cuda module to build gpu code
module load cudatoolkit

# Function to check last command's success
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

# Load modules
module load python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1 cray-mpich

# Ensure Conda is available
type conda >/dev/null 2>&1 || { echo "conda command not found. Please install Anaconda/Miniconda first."; exit 1; }

echo "Enter the name for the new conda environment (e.g., firm3d-simsopt):"
read -p "Your input: " env_name

echo "Creating conda environment: $env_name"
conda create -n "$env_name" --clone nersc-python
check_success "Failed to create conda environment $env_name"

echo "Activating conda environment: $env_name"
source activate "$env_name" || conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"

# FIRM3D Installation
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
env CC=cc CXX=CC pip install -e .
check_success "Failed to install FIRM3D"
cd ..

# Install simsopt (and upgrade numpy>=2.0 for jax compatibility)
pip install "numpy>=2.0" simsopt
check_success "Failed to install simsopt"

echo "Successfully installed FIRM3D + simsopt into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"
