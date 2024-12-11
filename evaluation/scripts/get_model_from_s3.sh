#!/bin/bash

# # Set variables (same values as in your original script)
# PREFIX="flim/Avengers-Bamba-9B-HF"
# LOCAL_PATH="/dccstor/eval-research/models/Avengers-Bamba-9B-HF"
# BUCKET_NAME="ibm-llm-input"
# ENDPOINT_URL="https://s3.us-east.cloud-object-storage.appdomain.cloud"

# Set variables (same values as in your original script)
PREFIX="Bamba-9B-2.65T"
LOCAL_PATH="/dccstor/fme/users/yotam/models/Bamba-9B-2.65T"
BUCKET_NAME="platform-vela-data"
ENDPOINT_URL="https://s3.us-east.cloud-object-storage.appdomain.cloud"

# Call the Python script with the arguments
.venv/bin/python evaluation/get_model_from_s3.py \
  --prefix "$PREFIX" \
  --local_path "$LOCAL_PATH" \
  --bucket_name "$BUCKET_NAME" \
  --endpoint_url "$ENDPOINT_URL"