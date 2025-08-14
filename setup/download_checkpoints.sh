#!/bin/bash

# Define the Google Drive folder URL and the target directory
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1MY4qcccN8Z31mWHCT06FittBlIFBRLFk?usp=sharing"
TARGET_DIR="checkpoints"

# Create the target directory if it doesn't exist
echo "Creating directory '$TARGET_DIR' if it doesn't exist..."
mkdir -p "$TARGET_DIR"

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "--------------------------------------------------------------------"
    echo "Error: 'gdown' command not found."
    echo "This script requires 'gdown' to download from Google Drive."
    echo "Please install it, typically via pip:"
    echo "  pip install gdown"
    echo "Or, manually download the files from the link into the '$TARGET_DIR' directory."
    echo "$GDRIVE_FOLDER_URL"
    echo "--------------------------------------------------------------------"
    exit 1
fi

# Download the folder contents into the target directory
echo "Attempting to download folder contents using gdown..."
echo "URL: $GDRIVE_FOLDER_URL"
echo "Target Directory: $TARGET_DIR"

# Use gdown to download the folder contents.
# The '-O' flag specifies the output directory where contents should be placed.
gdown --folder "$GDRIVE_FOLDER_URL" -O "$TARGET_DIR/" --quiet --fuzzy

# Check the exit status of gdown
if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------------------"
    echo "Successfully downloaded contents to '$TARGET_DIR'."
    echo "Files in '$TARGET_DIR':"
    ls -l "$TARGET_DIR"
    echo "--------------------------------------------------------------------"
else
    echo "--------------------------------------------------------------------"
    echo "Error: gdown failed to download the folder contents."
    echo "Please check the URL, your permissions, or try downloading manually."
    echo "$GDRIVE_FOLDER_URL"
    echo "--------------------------------------------------------------------"
    exit 1
fi

exit 0