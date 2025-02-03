#!/bin/bash

# Print Jupyter Notebook version
jupyter --version

# Create necessary directories if they don't exist
mkdir -p /home/jovyan/work
mkdir -p /home/jovyan/.jupyter/custom

# Default notebook file path
NOTEBOOK_PATH="/home/jovyan/work/example.ipynb"

# Check if NOTEBOOK_URL or NOTEBOOK_JSON is set
if [ ! -f "$NOTEBOOK_PATH" ]; then
    if [ ! -z "$NOTEBOOK_URL" ]; then
        echo "Downloading notebook from $NOTEBOOK_URL..."
        wget -O $NOTEBOOK_PATH $NOTEBOOK_URL
        echo "Notebook downloaded successfully."
    elif [ ! -z "$NOTEBOOK_JSON" ]; then
        echo "Saving notebook JSON to $NOTEBOOK_PATH..."
        echo "$NOTEBOOK_JSON" > $NOTEBOOK_PATH
        echo "Notebook saved successfully."
    else
        echo "No notebook provided. Using default or existing notebook."
    fi
else
    echo "Notebook already exists at $NOTEBOOK_PATH."
fi

# Set the Jupyter token
TOKEN=${JUPYTER_TOKEN:-'mytoken'}
echo "Starting Jupyter Notebook with token: $TOKEN"

# Start Flask API in the background
python /home/jovyan/notebook_api.py &

# Start Jupyter Notebook
exec jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token="$TOKEN" \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.allow_remote_access=True \
    --NotebookApp.disable_check_xsrf=True \
    --NotebookApp.tornado_settings='{"headers": {"Content-Security-Policy": "frame-ancestors *", "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}}'
