#!/bin/bash

# Automated Training Script for Background Execution
# Usage: ./auto_train.sh [model_type] [days] [options]
# Example: ./auto_train.sh timesfm 180
# Example: ./auto_train.sh lstm 365 "--epochs 100 --batch-size 32"

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory
mkdir -p "$LOG_DIR"

# Parse arguments
MODEL_TYPE=${1:-timesfm}
DAYS=${2:-180}
EXTRA_ARGS=${3:-""}

# Log file
LOG_FILE="$LOG_DIR/training_${MODEL_TYPE}_${TIMESTAMP}.log"

echo "=========================================="
echo "Automated BTC Prediction Training"
echo "=========================================="
echo "Model: $MODEL_TYPE"
echo "Days: $DAYS"
echo "Extra args: $EXTRA_ARGS"
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo "=========================================="

# Function to log and print
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Activate virtual environment
log "Activating virtual environment..."
source "$PROJECT_DIR/venv/bin/activate"

# Navigate to project directory
cd "$PROJECT_DIR"

# Run training based on model type
log "Starting training for $MODEL_TYPE..."

case $MODEL_TYPE in
    timesfm)
        log "Running TimesFM evaluation (pre-trained model)..."
        python src/train.py --model timesfm --days $DAYS $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
        ;;

    lstm)
        log "Training LSTM model (this may take several hours)..."
        python src/train.py --model lstm --days $DAYS $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
        ;;

    gru)
        log "Training GRU model..."
        python src/train.py --model gru --days $DAYS $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
        ;;

    xgboost)
        log "Training XGBoost model..."
        python src/train.py --model xgboost --days $DAYS $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
        ;;

    all)
        log "Training ALL models (this will take many hours)..."
        python src/train.py --model all --days $DAYS $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
        ;;

    *)
        log "ERROR: Unknown model type: $MODEL_TYPE"
        log "Valid options: timesfm, lstm, gru, xgboost, all"
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "✓ Training completed successfully!"
    log "Results saved in: results/"
    log "Models saved in: models/trained/"

    # Show latest results
    log "Latest metrics:"
    latest_metrics=$(ls -t "$PROJECT_DIR/results/"*_metrics_*.json 2>/dev/null | head -1)
    if [ -f "$latest_metrics" ]; then
        cat "$latest_metrics" | tee -a "$LOG_FILE"
    fi
else
    log "✗ Training failed with exit code: $EXIT_CODE"
fi

log "Finished: $(date)"
log "Total log file: $LOG_FILE"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "View log: cat $LOG_FILE"
echo "View results: ls -lh results/"
echo "View models: ls -lh models/trained/"
echo "=========================================="

exit $EXIT_CODE
