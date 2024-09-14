#!/bin/bash

# Default notebook file path
NOTEBOOK_PATH="/home/jovyan/work/example.ipynb"

# Check if the NOTEBOOK_URL or NOTEBOOK_JSON environment variables are set
if [ ! -f "$NOTEBOOK_PATH" ]; then
    if [ ! -z "$NOTEBOOK_URL" ]; then
        echo "Downloading notebook from $NOTEBOOK_URL..."
        wget -O $NOTEBOOK_PATH $NOTEBOOK_URL
    elif [ ! -z "$NOTEBOOK_JSON" ]; then
        echo "Saving notebook JSON to $NOTEBOOK_PATH..."
        echo "$NOTEBOOK_JSON" > $NOTEBOOK_PATH
    else
        echo "No notebook provided. Using default or existing notebook."
    fi
else
    echo "Notebook already exists at $NOTEBOOK_PATH."
fi

# Start JupyterLab and open the notebook directly, with extra flags to allow iframes and disable CSRF checks
exec jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='' \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.trust_xheaders=True \
    --NotebookApp.disable_check_xsrf=True \
    --NotebookApp.tornado_settings='{"headers":{"Content-Security-Policy":"frame-ancestors *"}}' \
    --LabApp.allow_remote_access=True
