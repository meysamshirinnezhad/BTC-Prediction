# Python Version Compatibility Fix

## Issue
TimesFM 2.5+ requires Python 3.11 or higher, but your system has Python 3.10.12.

## Solution Options

### Option 1: Install Python 3.11 (Recommended)

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version  # Should show Python 3.11.x

# Navigate to your project
cd ~/BTC-Prediction

# Remove old venv
deactivate  # If currently in venv
rm -rf venv

# Create new venv with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Verify Python version in venv
python --version  # Should show 3.11.x

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Use TimesFM 1.3.0 (Older Version, Python 3.10 Compatible)

If you can't upgrade Python, use the older TimesFM version:

```bash
cd ~/BTC-Prediction
source venv/bin/activate

# Install older TimesFM version
pip install timesfm==1.3.0

# Note: This version has fewer features than 2.5
```

### Option 3: Use Docker (Isolation, Any Python Version)

```bash
# Create Dockerfile in ~/BTC-Prediction
cat > ~/BTC-Prediction/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

CMD ["/bin/bash"]
EOF

# Build image
cd ~/BTC-Prediction
docker build -t btc-prediction .

# Run container
docker run -it --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models btc-prediction
```

## Recommended Steps (Option 1)

```bash
# 1. Clean up old installation
cd ~
rm -rf timesfm  # Remove cloned repo
cd ~/BTC-Prediction
deactivate || true
rm -rf venv

# 2. Install Python 3.11
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 3. Recreate environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify TimesFM works
python -c "import timesfm; print('TimesFM version:', timesfm.__version__)"

# 6. Run quick test
./quick_start.sh  # Choose option 1 (TimesFM demo)
```

## If Installation Fails

### Problem: `add-apt-repository` not found
```bash
sudo apt install -y software-properties-common
```

### Problem: PPA doesn't work on your distro
Build from source:
```bash
cd ~
wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
tar -xzf Python-3.11.7.tgz
cd Python-3.11.7
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall  # Use altinstall to not override system Python

# Verify
python3.11 --version
```

### Problem: Can't install Python 3.11
Use TimesFM 1.3.0 instead (works with Python 3.10):
```bash
# Edit requirements.txt
sed -i 's/timesfm>=2.5.0/timesfm==1.3.0/' requirements.txt

# Reinstall
pip install -r requirements.txt
```

## After Fixing

Test that everything works:
```bash
source venv/bin/activate
python -c "import timesfm; import torch; print('All good!')"

# Quick test
./auto_train.sh timesfm 30  # Small test with 30 days
```

## What Changes for TimesFM 1.3.0 vs 2.5.0?

If you use the older version, some features differ:

| Feature | TimesFM 1.3.0 | TimesFM 2.5.0 |
|---------|---------------|---------------|
| Max Context | 512 | 16,384 |
| Model Size | 500M params | 200M params |
| Quantile Forecast | ✓ | ✓ Enhanced |
| Speed | Slower | Faster |

The code will still work, just with less capacity.
