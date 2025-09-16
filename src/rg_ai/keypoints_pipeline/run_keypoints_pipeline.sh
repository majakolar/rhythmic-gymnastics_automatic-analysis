#!/bin/bash

set -e  
set -u  

CONFIG_PATH="src/rg_ai/keypoints_pipeline/config.yaml"
LOG_DIR="logs/keypoints_pipeline"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

handle_error() {
    local exit_code=$?
    local step=$1
    log "ERROR: ${step} failed with exit code ${exit_code}"
    exit $exit_code
}

if [ ! -f "$CONFIG_PATH" ]; then
    log "ERROR: Configuration file not found at $CONFIG_PATH"
    exit 1
fi


log "Starting keypoints pipeline with config: $CONFIG_PATH"

# Step 1: Extract Keypoints
log "Step 1/3: Starting keypoints extraction..."
python src/rg_ai/keypoints_pipeline/extract_keypoints.py -c "$CONFIG_PATH" || handle_error "Keypoints extraction"
log "Keypoints extraction completed successfully"

# Step 2: Make 3D Embeddings
log "Step 2/3: Starting 3D embeddings generation..."
python src/rg_ai/keypoints_pipeline/make_3d_embeddings.py -c "$CONFIG_PATH"  || handle_error "3D embeddings generation"
log "3D embeddings generation completed successfully." 

# Step 3: Movement Classification
log "Step 3/3: Starting movement classification..."
python src/rg_ai/keypoints_pipeline/movement_classification.py --config "$CONFIG_PATH"  || handle_error "Movement classification"
log "Movement classification completed successfully"

log "Pipeline completed successfully!"

# Print path to log file
echo "Pipeline log saved to: $LOG_FILE"
