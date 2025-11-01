#!/bin/bash

# Julius Baer AML Monitoring System - Setup Script
# This script automates the setup process for new users

set -e  # Exit on any error

echo "🏦 Julius Baer AML Monitoring System - Setup Script"
echo "=================================================="

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8 or higher is required. Current version: $(python --version)"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python version: $(python --version)"

# Detect operating system
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "🖥️  Detected OS: ${MACHINE}"

# Install system dependencies
echo "📦 Installing system dependencies..."

if [[ "$MACHINE" == "Mac" ]]; then
    echo "🍺 Installing macOS dependencies with Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies
    echo "Installing OpenMP and Tesseract..."
    brew install libomp tesseract
    
elif [[ "$MACHINE" == "Linux" ]]; then
    echo "🐧 Installing Linux dependencies..."
    
    # Detect Linux distribution
    if [[ -f /etc/debian_version ]]; then
        echo "Detected Debian/Ubuntu system"
        sudo apt update
        sudo apt install -y libomp-dev build-essential tesseract-ocr tesseract-ocr-eng \
                           libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
    elif [[ -f /etc/redhat-release ]]; then
        echo "Detected Red Hat/CentOS system"
        sudo yum install -y gcc gcc-c++ make tesseract tesseract-langpack-eng
    else
        echo "⚠️  Unknown Linux distribution. Please install OpenMP and Tesseract manually."
    fi
    
else
    echo "⚠️  For Windows, please install:"
    echo "   - Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki"
    echo "   - Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe"
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [[ -d ".venv" ]]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf .venv
fi

python -m venv .venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$MACHINE" == "Mac" ]] || [[ "$MACHINE" == "Linux" ]]; then
    source .venv/bin/activate
else
    .venv/Scripts/activate
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Verify installation
echo "🧪 Testing installation..."
python test_optimized_model.py

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 SUCCESS! Setup completed successfully!"
    echo "=================================================="
    echo ""
    echo "🚀 To start the application:"
    echo "   1. Activate the virtual environment:"
    if [[ "$MACHINE" == "Mac" ]] || [[ "$MACHINE" == "Linux" ]]; then
        echo "      source .venv/bin/activate"
    else
        echo "      .venv\\Scripts\\activate"
    fi
    echo "   2. Run the application:"
    echo "      streamlit run src/frontend/app.py"
    echo ""
    echo "📱 The app will be available at: http://localhost:8501"
    echo ""
else
    echo "❌ Installation test failed. Please check the error messages above."
    exit 1
fi