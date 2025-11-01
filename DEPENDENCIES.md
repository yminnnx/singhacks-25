# Dependencies Overview

## Python Dependencies (requirements.txt)

### Core Application Framework
- **streamlit==1.28.1** - Web application framework for the dashboard
- **fastapi==0.104.1** - REST API framework (for future API endpoints)
- **uvicorn==0.24.0** - ASGI server for FastAPI

### Data Processing & Analysis
- **pandas==2.1.3** - Data manipulation and analysis
- **numpy==1.24.3** - Numerical computing library

### Machine Learning
- **scikit-learn==1.3.2** - Machine learning library
- **xgboost>=2.0.0** - **REQUIRED for optimized ML model**
- **joblib>=1.3.0** - Model serialization and parallel processing

### Visualization
- **plotly==5.17.0** - Interactive charts and graphs
- **matplotlib==3.8.2** - Static plotting library
- **seaborn==0.13.0** - Statistical visualization

### Document Processing
- **PyPDF2==3.0.1** - PDF file processing
- **python-docx==1.1.0** - Word document processing
- **pytesseract==0.3.10** - OCR text extraction

### Image Processing
- **Pillow==10.1.0** - Image processing library
- **opencv-python==4.8.1.78** - Computer vision library

### AI & NLP
- **openai==1.3.5** - OpenAI API integration
- **langchain==0.0.339** - LLM application framework
- **langchain-openai==0.0.2** - LangChain OpenAI integration
- **sentence-transformers==2.2.2** - Sentence embeddings
- **chromadb==0.4.18** - Vector database

### Web & API
- **requests==2.31.0** - HTTP library
- **beautifulsoup4==4.12.2** - HTML/XML parsing
- **python-multipart==0.0.6** - Multipart form data parsing
- **aiofiles==23.2.1** - Async file operations

### Utilities
- **python-dateutil==2.8.2** - Date/time utilities
- **jinja2==3.1.2** - Template engine
- **watchdog>=3.0.0** - File system monitoring
- **psutil>=5.9.0** - System monitoring

### Security & Authentication
- **python-jose[cryptography]==3.3.0** - JWT token handling
- **passlib[bcrypt]==1.7.4** - Password hashing
- **pydantic==2.5.0** - Data validation

### Database
- **sqlalchemy==2.0.23** - SQL toolkit and ORM
- **alembic==1.12.1** - Database migrations

## System Dependencies

### Required for All Platforms
- **Python 3.8+** - Programming language runtime
- **pip** - Python package installer

### macOS Specific
```bash
brew install libomp        # OpenMP for XGBoost
brew install tesseract     # OCR engine
```

### Ubuntu/Debian Linux
```bash
sudo apt install libomp-dev build-essential tesseract-ocr tesseract-ocr-eng
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### Windows
- **Tesseract OCR**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Visual C++ Redistributable**: Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## Critical Dependencies for ML Model

The optimized XGBoost model requires these specific components:

1. **xgboost>=2.0.0** - The ML model algorithm
2. **OpenMP runtime** - Parallel processing support
   - macOS: `libomp.dylib` (installed via Homebrew)
   - Linux: `libgomp.so` (usually included)
   - Windows: Included with Visual C++ Redistributable
3. **joblib>=1.3.0** - Model serialization
4. **scikit-learn==1.3.2** - Feature preprocessing and label encoding

## Installation Verification

After installing dependencies, verify with:
```bash
python test_optimized_model.py
```

Expected output:
```
✅ Optimized ML Model loaded successfully!
🎯 OPTIMIZED MODEL DETECTED!
   High recall (97.4%) indicates optimization for better detection
```

## Troubleshooting Dependencies

### XGBoost Issues
- **macOS**: `brew install libomp`
- **Linux**: `sudo apt install libgomp1`
- **Windows**: Install Visual C++ Redistributable

### Tesseract Issues
- Add Tesseract to system PATH
- Verify installation: `tesseract --version`

### OpenCV Issues
- Try: `pip install opencv-python-headless` instead of `opencv-python`

### Version Conflicts
- Use virtual environment to isolate dependencies
- Update pip: `pip install --upgrade pip`
- Clear pip cache: `pip cache purge`