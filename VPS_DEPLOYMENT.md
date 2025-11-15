# VPS Deployment & Fine-Tuning Guide

Complete guide for deploying and fine-tuning the BTC prediction system on your VPS.

## Table of Contents
- [Initial Setup](#initial-setup)
- [Installation](#installation)
- [Running Models](#running-models)
- [Fine-Tuning](#fine-tuning)
- [GPU Setup](#gpu-setup)
- [Production Deployment](#production-deployment)
- [Monitoring & Automation](#monitoring--automation)

---

## Initial Setup

### 1. System Requirements

**Minimum Requirements:**
- RAM: 4GB (8GB+ recommended for deep learning)
- Storage: 10GB free space
- CPU: 2+ cores
- OS: Ubuntu 20.04+ / Debian 11+ / CentOS 8+

**Recommended for Deep Learning:**
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM (optional but speeds up training)
- Storage: 20GB+ SSD

### 2. Connect to VPS

```bash
ssh your-username@your-vps-ip
```

### 3. Update System

```bash
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y git python3 python3-pip python3-venv build-essential
```

### 4. Clone Repository

```bash
cd ~
git clone https://github.com/meysamshirinnezhad/BTC-Prediction.git
cd BTC-Prediction
```

---

## Installation

### Option 1: CPU-Only Installation (Faster, No GPU)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# If you get errors with TA-Lib, install system dependencies first:
sudo apt install -y ta-lib libta-lib-dev
pip install TA-Lib
```

### Option 2: GPU Installation (NVIDIA GPU)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### Option 3: Minimal Installation (TimesFM Only)

If you only want to use TimesFM (no training needed):

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy pandas yfinance matplotlib seaborn
pip install timesfm[torch]
```

---

## Running Models

### 1. Quick Start - TimesFM (No Training!)

TimesFM is pre-trained and ready to use immediately:

```bash
# Activate environment
source venv/bin/activate

# Run TimesFM evaluation (fast, no training)
python src/train.py --model timesfm --days 180 --context-length 512

# Make predictions
python src/predict.py \
    --model models/trained/timesfm_config_*.pkl \
    --model-type timesfm \
    --horizon 24 \
    --plot
```

**Advantages:**
- ✅ No training time (seconds instead of hours)
- ✅ Works on CPU (no GPU needed)
- ✅ Built-in confidence intervals
- ✅ Good baseline performance

### 2. Train LSTM Model

```bash
source venv/bin/activate

# Basic training (2-4 hours on CPU, 30-60 min on GPU)
python src/train.py --model lstm --days 365 --epochs 100 --batch-size 32

# With custom sequence length
python src/train.py --model lstm --days 365 --epochs 100 --sequence-length 120

# Quick test run (10 epochs)
python src/train.py --model lstm --days 90 --epochs 10
```

### 3. Train GRU Model

```bash
source venv/bin/activate

# GRU is faster than LSTM
python src/train.py --model gru --days 365 --epochs 100 --batch-size 32
```

### 4. Train XGBoost Model

```bash
source venv/bin/activate

# XGBoost is fastest to train (10-30 minutes)
python src/train.py --model xgboost --days 365
```

### 5. Train All Models

```bash
source venv/bin/activate

# This will take several hours
python src/train.py --model all --days 365 --epochs 100
```

---

## Fine-Tuning

### LSTM/GRU Fine-Tuning

#### 1. Hyperparameter Tuning

Edit `src/config.py` or use command-line arguments:

```bash
# Experiment with different architectures
python src/train.py --model lstm --days 365 --epochs 100 --sequence-length 60
python src/train.py --model lstm --days 365 --epochs 100 --sequence-length 120
python src/train.py --model lstm --days 365 --epochs 100 --sequence-length 180

# Try different batch sizes
python src/train.py --model lstm --days 365 --epochs 100 --batch-size 16
python src/train.py --model lstm --days 365 --epochs 100 --batch-size 64
```

#### 2. Modify Model Architecture

Edit the training function in `src/train.py`:

```python
# Find this line in train_lstm_model():
model = create_lstm_model(
    sequence_length=data['sequence_length'],
    n_features=data['n_features'],
    model_type=model_type,
    lstm_units=[128, 64, 32],  # <- MODIFY THIS
    dropout_rate=0.2,           # <- AND THIS
    learning_rate=0.001         # <- AND THIS
)
```

**Tuning Options:**

```python
# Deeper network (more layers)
lstm_units=[256, 128, 64, 32]

# Wider network (more units per layer)
lstm_units=[256, 128, 64]

# Smaller network (faster training)
lstm_units=[64, 32]

# Adjust dropout (prevent overfitting)
dropout_rate=0.3  # More regularization
dropout_rate=0.1  # Less regularization

# Adjust learning rate
learning_rate=0.0001  # Slower, more stable
learning_rate=0.01    # Faster, less stable
```

#### 3. Advanced Configuration

Create a custom training script:

```python
# my_custom_training.py
import sys
sys.path.append('.')

from src.data_collector import BTCDataCollector
from src.feature_engineering import create_features
from src.preprocessing import prepare_data_for_lstm
from models.lstm_model import create_lstm_model

# Collect more data
collector = BTCDataCollector()
df_raw = collector.fetch_yahoo_finance(days=730, interval='1h')  # 2 years

# Create features
df = create_features(df_raw, feature_set='all')

# Prepare data with custom split
data = prepare_data_for_lstm(
    df,
    sequence_length=180,      # Longer sequences
    forecast_horizon=1,
    train_ratio=0.8,          # More training data
    val_ratio=0.1,
    test_ratio=0.1
)

# Create custom model
model = create_lstm_model(
    sequence_length=180,
    n_features=data['n_features'],
    lstm_units=[256, 128, 64, 32],  # Deeper network
    dropout_rate=0.25,
    learning_rate=0.0005
)

model.build_model()

# Train with more epochs
history = model.train(
    data['X_train'],
    data['y_train'],
    data['X_val'],
    data['y_val'],
    epochs=200,              # More epochs
    batch_size=32
)

# Save model
model.save_model('models/trained/my_custom_lstm.h5')
data['preprocessor'].save_scalers('models/trained/my_custom_preprocessor.pkl')
```

Run it:
```bash
python my_custom_training.py
```

### XGBoost Fine-Tuning

Edit `src/train.py` in the `train_xgboost_model()` function:

```python
model = create_xgboost_model(
    n_estimators=2000,      # More trees (default 1000)
    max_depth=10,           # Deeper trees (default 7)
    learning_rate=0.005,    # Slower learning (default 0.01)
    subsample=0.9,          # More data per tree
    colsample_bytree=0.9    # More features per tree
)
```

### TimesFM Fine-Tuning

TimesFM is pre-trained, but you can adjust its configuration:

```python
from models.timesfm_model import create_timesfm_model

# Longer context for more historical data
model = create_timesfm_model(
    backend='pytorch',
    max_context=2048,        # Increase from 512-1024
    max_horizon=512,         # Longer prediction horizon
    normalize_inputs=True,
    use_continuous_quantile_head=True
)
```

Or via command line:
```bash
python src/train.py --model timesfm --context-length 2048 --days 730
```

---

## GPU Setup

### Check GPU Availability

```bash
# Check if NVIDIA GPU is present
nvidia-smi

# Check PyTorch GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

### Install NVIDIA Drivers (if needed)

```bash
# Ubuntu/Debian
sudo apt install -y nvidia-driver-525 nvidia-cuda-toolkit

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### GPU Memory Management

If you run out of GPU memory:

```bash
# Reduce batch size
python src/train.py --model lstm --batch-size 16  # or 8

# Use smaller model
# Edit lstm_units in train.py to [64, 32] instead of [128, 64, 32]
```

---

## Production Deployment

### 1. Screen/Tmux for Long Training Sessions

Training can take hours. Use screen or tmux to keep it running:

```bash
# Install screen
sudo apt install -y screen

# Start a new screen session
screen -S btc-training

# Run training
source venv/bin/activate
python src/train.py --model all --days 365 --epochs 100

# Detach: Press Ctrl+A, then D

# Reattach later
screen -r btc-training

# List sessions
screen -ls
```

Or use tmux:
```bash
sudo apt install -y tmux

# Start session
tmux new -s btc-training

# Run training
source venv/bin/activate
python src/train.py --model all --days 365 --epochs 100

# Detach: Press Ctrl+B, then D
# Reattach: tmux attach -t btc-training
```

### 2. Automated Training Script

Create `run_training.sh`:

```bash
#!/bin/bash

# Activate environment
source ~/BTC-Prediction/venv/bin/activate

# Navigate to project
cd ~/BTC-Prediction

# Run training with error handling
python src/train.py --model timesfm --days 365 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Send notification (optional)
echo "Training completed at $(date)" | mail -s "BTC Training Complete" your@email.com
```

Make it executable:
```bash
chmod +x run_training.sh
./run_training.sh
```

### 3. Schedule Regular Retraining (Cron)

```bash
# Create logs directory
mkdir -p ~/BTC-Prediction/logs

# Edit crontab
crontab -e

# Add this line to retrain every week on Sunday at 2 AM:
0 2 * * 0 /home/your-username/BTC-Prediction/run_training.sh
```

### 4. Background Training with nohup

```bash
# Run in background
nohup python src/train.py --model lstm --days 365 --epochs 100 > training.log 2>&1 &

# Check progress
tail -f training.log

# Find process
ps aux | grep train.py

# Kill if needed
kill <PID>
```

---

## Monitoring & Automation

### 1. Monitor Training Progress

Check results directory:
```bash
ls -lh results/
cat results/lstm_metrics_*.json
```

### 2. Track GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or install htop for CPU monitoring
sudo apt install -y htop
htop
```

### 3. Disk Space Management

```bash
# Check disk usage
df -h

# Clean up old data
rm data/raw/btc_data_old_*.csv

# Clean old models (keep only latest)
ls -t models/trained/*.h5 | tail -n +6 | xargs rm  # Keep 5 newest
```

### 4. Backup Trained Models

```bash
# Create backup script
cat > backup_models.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf btc_models_backup_$DATE.tar.gz models/trained/
# Upload to cloud storage or external server
# scp btc_models_backup_$DATE.tar.gz user@backup-server:/backups/
EOF

chmod +x backup_models.sh
```

---

## Quick Reference Commands

### Start Training (Recommended Order)

```bash
# 1. Start with TimesFM (fast baseline)
python src/train.py --model timesfm --days 180

# 2. Then XGBoost (medium speed)
python src/train.py --model xgboost --days 365

# 3. Then GRU (faster than LSTM)
python src/train.py --model gru --days 365 --epochs 50

# 4. Finally LSTM (slowest but often best)
python src/train.py --model lstm --days 365 --epochs 100
```

### Make Predictions

```bash
# TimesFM (fastest)
python src/predict.py --model-type timesfm --horizon 24

# LSTM (if trained)
python src/predict.py \
    --model models/trained/lstm_model_*.h5 \
    --model-type lstm \
    --preprocessor models/trained/lstm_preprocessor_*.pkl \
    --horizon 24 \
    --plot
```

### Compare All Models

```bash
# After training all models, check results
cat results/*_metrics_*.json | jq '.metrics'
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python src/train.py --model lstm --batch-size 8

# Use smaller context
python src/train.py --model timesfm --context-length 256

# Free up memory
sync; echo 3 > /proc/sys/vm/drop_caches
```

### Training Too Slow

```bash
# Use fewer epochs for testing
python src/train.py --model lstm --epochs 10 --days 90

# Use smaller dataset
python src/train.py --model lstm --days 180  # instead of 365

# Use GRU instead of LSTM (faster)
python src/train.py --model gru --days 365
```

### Dependencies Issues

```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# If TA-Lib fails
sudo apt install -y ta-lib libta-lib-dev
pip install --upgrade TA-Lib
```

---

## Performance Optimization Tips

1. **Start Simple**: Begin with TimesFM or XGBoost for quick results
2. **Use GPU**: Deep learning models (LSTM/GRU) are 10-50x faster on GPU
3. **Batch Training**: Train multiple configurations in sequence overnight
4. **Save Checkpoints**: Models auto-save, so you won't lose progress
5. **Monitor Resources**: Use `htop` and `nvidia-smi` to avoid OOM
6. **Cloud Burst**: For heavy training, consider cloud GPU (AWS, GCP, Paperspace)

---

## Next Steps

1. **Run TimesFM first** - Get baseline in minutes
2. **Train XGBoost** - Fast and interpretable
3. **Train LSTM/GRU** - Best performance (if you have time/GPU)
4. **Compare results** - Check `results/` directory
5. **Fine-tune best model** - Adjust hyperparameters
6. **Deploy for predictions** - Set up automated retraining

Good luck with your VPS deployment! 🚀
