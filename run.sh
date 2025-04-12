#!/bin/bash

sudo apt install python3.12-venv python3-pip python3-full unzip

if [ ! -f "logs" ]; then
    # Create Data directory if it doesn't exist
    mkdir -p "logs"
fi

if [ ! -f "Models" ]; then
  # Create Data directory if it doesn't exist
  mkdir -p "Models"
fi

if [ ! -f "results" ]; then
  # Create Data directory if it doesn't exist
  mkdir -p "results"
fi


# Check if python environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating python virtual environment..."
    python3 -m venv venv
fi

# Activate the environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed, if not install them
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
fi

echo ""

# DATASET DOWNLOAD / EXTRACTION
DATA_DIR="Data"
TARGET_FILE="${DATA_DIR}/papers.SSN.jsonl"
# shellcheck disable=SC2034
ZIP_FILE="${DATA_DIR}/SSN.zip"

# Check if the target file already exists
if [ ! -f "${TARGET_FILE}" ]; then
    # Create Data directory if it doesn't exist
    mkdir -p "${DATA_DIR}"
    
    cd "${DATA_DIR}" || exit

    # Download zip file only if it doesn't exist
    if [ ! -f "SSN.zip" ]; then
        echo "Downloading zip file..."
        python3 downloader.py || {
            echo "Failed to download file"
            exit 1
        }
    else
        echo "Zip file already exists. Skipping download."
    fi

    # Extract the entire zip file
    echo "Extracting zip file..."
    unzip -o "SSN.zip" || {
        echo "Failed to extract zip file"
        exit 1
    }
    
    # Move the files from SSN/ directory to Data/ directory
    echo "Moving files from SSN/ directory..."
    if [ -f "SSN/papers.SSN.jsonl" ]; then
        mv "SSN/papers.SSN.jsonl" .
        mv "SSN/citation_relations.json" . 2>/dev/null || true  # This file is optional
        rmdir "SSN"  # Remove the now-empty directory
    else
        echo "Error: papers.SSN.jsonl was not found in SSN/ directory!"
        echo "Contents of zip file:"
        unzip -l "SSN.zip"
        exit 1
    fi
    
    # Clean up zip file after extraction
    rm "SSN.zip"
    echo "Extraction complete and zip file removed."
    
    cd ..
else
    echo "Target file papers.SSN.jsonl already exists. Skipping download and extraction."
fi

echo "Data transforming and splitting"
python Data/Utils/transformer.py

echo "Running train"
python train.py

# Deactivate the virtual environment when done
deactivate
echo "Script completed."
