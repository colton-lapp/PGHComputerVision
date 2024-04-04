#!/bin/bash

# Define the name of the virtual environment directory
VENV_NAME="myenv"

# Define the path to the virtual environment activation script
VENV_ACTIVATE="$VENV_NAME/bin/activate"

# Check if the directory has a virtual environment and activate it
if [ -e "$PWD/$VENV_NAME/bin/activate" ]; then
    source "$PWD/$VENV_ACTIVATE"
    echo "Activated virtual environment: $VENV_NAME"
fi