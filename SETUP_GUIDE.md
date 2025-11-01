# Setup Guide for Julius Baer AML Monitoring System

## Prerequisites

### System Requirements
- Python 3.8 or higher
- macOS, Linux, or Windows
- At least 4GB RAM
- 2GB free disk space

### System-Level Dependencies

#### For macOS Users:
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenMP (required for XGBoost)
brew install libomp

# Install Tesseract (for OCR functionality)
brew install tesseract

# Optional: Install Xcode command line tools for better performance
xcode-select --install
```

#### For Ubuntu/Debian Linux Users:
```bash
# Update package list
sudo apt update

# Install OpenMP and build essentials
sudo apt install -y libomp-dev build-essential

# Install Tesseract for OCR
sudo apt install -y tesseract-ocr tesseract-ocr-eng

# Install additional system libraries
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

#### For Windows Users:
```powershell
# Install Tesseract OCR
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

# Install Microsoft Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Note: XGBoost should work out of the box on Windows
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd singhacks-25
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues with XGBoost on macOS, try:
pip install --upgrade pip
pip install xgboost --no-cache-dir
```

### 4. Verify Installation
```bash
# Test the optimized model integration
python test_optimized_model.py

# You should see:
# ✅ Optimized ML Model loaded successfully!
# 🎯 OPTIMIZED MODEL DETECTED!
```

### 5. Run the Application
```bash
# Start the Streamlit dashboard
streamlit run src/frontend/app.py

# Application will be available at: http://localhost:8501
```

## Troubleshooting

### Common Issues and Solutions

#### XGBoost Import Error on macOS
```bash
# Error: Library not loaded: @rpath/libomp.dylib
brew install libomp
```

#### Tesseract Not Found Error
```bash
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt install tesseract-ocr

# Windows: Download and install from GitHub link above
```

#### OpenCV Issues
```bash
# If OpenCV fails to import:
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Streamlit Page Config Error
```bash
# If you see "set_page_config() can only be called once":
# This has been fixed in the current version
# Make sure you're using the latest code
```

### Performance Optimization

#### For Better Streamlit Performance:
```bash
# Install watchdog for faster file watching
pip install watchdog

# Install Xcode command line tools (macOS)
xcode-select --install
```

#### Memory Optimization:
- Close other applications when running large transaction analyses
- Use smaller sample sizes for testing (adjust in the frontend)
- Monitor system resources during ML model predictions

## Verification Checklist

After installation, verify these components work:

- [ ] Python environment activated
- [ ] All packages installed without errors
- [ ] XGBoost model loads successfully
- [ ] Streamlit application starts
- [ ] Dashboard displays optimized model metrics
- [ ] Transaction analysis runs with ML predictions
- [ ] No critical errors in console

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all system dependencies are installed
3. Verify Python version compatibility (3.8+)
4. Check console output for specific error messages
5. Try running the test script: `python test_optimized_model.py`

## Quick Start Commands

For experienced users, here's the complete setup in one block:

```bash
# macOS/Linux Quick Setup
git clone <repository-url>
cd singhacks-25
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
brew install libomp tesseract  # macOS only
pip install -r requirements.txt
python test_optimized_model.py  # Verify installation
streamlit run src/frontend/app.py  # Start application
```

The application should now be running with the optimized ML model at http://localhost:8501