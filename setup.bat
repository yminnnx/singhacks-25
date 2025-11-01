@echo off
REM Julius Baer AML Monitoring System - Windows Setup Script

echo 🏦 Julius Baer AML Monitoring System - Windows Setup
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python version: 
python --version

REM Create virtual environment
echo 🐍 Creating Python virtual environment...
if exist .venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q .venv
)

python -m venv .venv

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo 📚 Installing Python packages...
pip install -r requirements.txt

REM Verify installation
echo 🧪 Testing installation...
python test_optimized_model.py

if %errorlevel% equ 0 (
    echo.
    echo 🎉 SUCCESS! Setup completed successfully!
    echo ==================================================
    echo.
    echo 🚀 To start the application:
    echo    1. Activate the virtual environment:
    echo       .venv\Scripts\activate
    echo    2. Run the application:
    echo       streamlit run src/frontend/app.py
    echo.
    echo 📱 The app will be available at: http://localhost:8501
    echo.
    echo ⚠️ Note: Make sure you have installed:
    echo    - Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
    echo    - Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
) else (
    echo ❌ Installation test failed. Please check the error messages above.
    pause
    exit /b 1
)

pause