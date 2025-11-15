#!/bin/bash

# Quick Start Script for BTC Prediction on VPS
# This script sets up and runs the BTC prediction system

set -e  # Exit on error

echo "=========================================="
echo "BTC Prediction Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_green() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_red() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_yellow "Virtual environment not found. Creating..."
    python3 -m venv venv
    print_green "Virtual environment created"
fi

# Activate virtual environment
print_yellow "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import pandas" 2>/dev/null; then
    print_yellow "Dependencies not installed. Installing..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_green "Dependencies installed"
else
    print_green "Dependencies already installed"
fi

# Create necessary directories
mkdir -p data/raw data/processed models/trained results logs

# Menu for user selection
echo ""
echo "=========================================="
echo "What would you like to do?"
echo "=========================================="
echo "1) Quick Demo - Run TimesFM (No training, ~5 min)"
echo "2) Train XGBoost Model (~30 min)"
echo "3) Train GRU Model (~2 hours)"
echo "4) Train LSTM Model (~4 hours)"
echo "5) Train ALL Models (~6+ hours)"
echo "6) Make Predictions"
echo "7) View Results"
echo "8) Exit"
echo ""
read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        print_yellow "Running TimesFM Quick Demo..."
        print_yellow "This will evaluate the pre-trained TimesFM model (no training needed)"

        read -p "How many days of data? (default: 180): " days
        days=${days:-180}

        read -p "Context length? (default: 512): " context
        context=${context:-512}

        print_yellow "Fetching data and running evaluation..."
        python src/train.py --model timesfm --days $days --context-length $context

        print_green "TimesFM evaluation complete!"
        print_yellow "Results saved in: results/"
        ;;

    2)
        print_yellow "Training XGBoost Model..."

        read -p "How many days of data? (default: 365): " days
        days=${days:-365}

        print_yellow "Starting training... This will take ~30 minutes"
        python src/train.py --model xgboost --days $days

        print_green "XGBoost training complete!"
        ;;

    3)
        print_yellow "Training GRU Model..."

        read -p "How many days of data? (default: 365): " days
        days=${days:-365}

        read -p "Number of epochs? (default: 100): " epochs
        epochs=${epochs:-100}

        read -p "Batch size? (default: 32): " batch
        batch=${batch:-32}

        print_yellow "Starting training... This will take ~2 hours"
        print_yellow "Tip: Use 'screen' or 'tmux' for long sessions"
        python src/train.py --model gru --days $days --epochs $epochs --batch-size $batch

        print_green "GRU training complete!"
        ;;

    4)
        print_yellow "Training LSTM Model..."

        read -p "How many days of data? (default: 365): " days
        days=${days:-365}

        read -p "Number of epochs? (default: 100): " epochs
        epochs=${epochs:-100}

        read -p "Batch size? (default: 32): " batch
        batch=${batch:-32}

        print_yellow "Starting training... This will take ~4 hours"
        print_yellow "Tip: Use 'screen' or 'tmux' for long sessions"
        python src/train.py --model lstm --days $days --epochs $epochs --batch-size $batch

        print_green "LSTM training complete!"
        ;;

    5)
        print_yellow "Training ALL Models..."
        print_red "WARNING: This will take 6+ hours!"

        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            print_yellow "Cancelled"
            exit 0
        fi

        read -p "How many days of data? (default: 365): " days
        days=${days:-365}

        read -p "Number of epochs for deep learning? (default: 100): " epochs
        epochs=${epochs:-100}

        print_yellow "Starting training... Grab a coffee (or several)!"
        python src/train.py --model all --days $days --epochs $epochs

        print_green "All models trained successfully!"
        ;;

    6)
        print_yellow "Make Predictions"
        echo ""
        echo "Available models:"
        ls -1 models/trained/ 2>/dev/null || echo "No models found. Train a model first."
        echo ""

        read -p "Model type (lstm/gru/xgboost/timesfm): " model_type

        if [ "$model_type" == "timesfm" ]; then
            # TimesFM doesn't need a trained model file
            read -p "Prediction horizon (hours)? (default: 24): " horizon
            horizon=${horizon:-24}

            python src/predict.py --model-type timesfm --horizon $horizon --plot
        else
            read -p "Model file path: " model_path
            read -p "Prediction horizon (hours)? (default: 24): " horizon
            horizon=${horizon:-24}

            if [ "$model_type" == "lstm" ] || [ "$model_type" == "gru" ]; then
                read -p "Preprocessor file path: " prep_path
                python src/predict.py --model "$model_path" --model-type $model_type \
                    --preprocessor "$prep_path" --horizon $horizon --plot
            else
                python src/predict.py --model "$model_path" --model-type $model_type \
                    --horizon $horizon --plot
            fi
        fi

        print_green "Predictions complete!"
        ;;

    7)
        print_yellow "Viewing Results..."
        echo ""

        if [ -d "results" ] && [ "$(ls -A results)" ]; then
            echo "Recent Results:"
            ls -lth results/ | head -10
            echo ""

            read -p "View a specific result file? (y/n): " view
            if [ "$view" == "y" ]; then
                read -p "Filename: " filename
                if [ -f "results/$filename" ]; then
                    cat "results/$filename"
                else
                    print_red "File not found"
                fi
            fi
        else
            print_yellow "No results found. Run training or predictions first."
        fi
        ;;

    8)
        print_yellow "Goodbye!"
        exit 0
        ;;

    *)
        print_red "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_green "Done!"
echo ""
echo "=========================================="
echo "What's Next?"
echo "=========================================="
echo "• View results: cat results/*_metrics_*.json"
echo "• Compare models: ls -lh models/trained/"
echo "• Make predictions: ./quick_start.sh (choose option 6)"
echo "• Read docs: cat VPS_DEPLOYMENT.md"
echo "=========================================="
