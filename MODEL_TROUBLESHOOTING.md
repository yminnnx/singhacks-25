# Model Loading Troubleshooting Guide

## Common Reasons Why the Optimized Model Falls Back to Rule-Based

### 🔍 **Quick Diagnosis**

Run this command on your other computer to diagnose the issue:

```bash
python diagnose_model.py
```

This will check all dependencies and provide specific solutions.

---

## 🚨 **Most Common Issues & Solutions**

### 1. **XGBoost Not Installed or Import Error**

**Symptoms:**

- Error: `No module named 'xgboost'`
- Error: `Library not loaded: libomp.dylib`

**Solutions:**

#### For macOS:

```bash
# Install OpenMP (required for XGBoost)
brew install libomp

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost==2.0.3 --no-cache-dir
```

#### For Linux (Ubuntu/Debian):

```bash
# Install OpenMP and build tools
sudo apt update
sudo apt install libgomp1 build-essential

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost==2.0.3 --no-cache-dir
```

#### For Windows:

```powershell
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost==2.0.3 --no-cache-dir
```

### 2. **Scikit-learn Version Mismatch**

**Symptoms:**

- Warning: `InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.3.0 when using version 1.7.2`
- Model fails to load properly

**Solution:**

```bash
# Install exact scikit-learn version used to train the model
pip install scikit-learn==1.3.2
```

### 3. **Missing Model Files**

**Symptoms:**

- No specific error, just falls back to rule-based
- Console shows: "Could not find model at..."

**Solution:**

```bash
# Ensure model files are present
ls -la models/
# Should show: aml_risk_model_optimized.pkl

# If missing, make sure you copied the models/ directory
```

### 4. **Virtual Environment Issues**

**Symptoms:**

- Dependencies seem installed but model still doesn't load
- Different behavior between computers

**Solution:**

```bash
# Ensure you're in the correct virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Verify Python path
which python
# Should point to .venv/bin/python

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### 5. **Python Version Compatibility**

**Symptoms:**

- Pickle loading errors
- Unexpected import failures

**Solution:**

```bash
# Check Python version
python --version
# Should be 3.8 or higher

# If using different Python versions, recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🛠️ **Step-by-Step Fix Process**

### Step 1: Run Diagnostic

```bash
python diagnose_model.py
```

### Step 2: Install System Dependencies

#### macOS:

```bash
brew install libomp tesseract
```

#### Linux:

```bash
sudo apt install libgomp1 build-essential tesseract-ocr
```

#### Windows:

- Install Visual C++ Redistributable
- Install Tesseract OCR

### Step 3: Fix Python Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Update pip
pip install --upgrade pip

# Install exact versions
pip install -r requirements.txt --force-reinstall
```

### Step 4: Test Model Loading

```bash
python test_optimized_model.py
```

Expected output:

```
✅ Optimized ML Model loaded successfully!
🎯 OPTIMIZED MODEL DETECTED!
```

### Step 5: Start Application

```bash
streamlit run src/frontend/app.py
```

---

## 🎯 **Quick Fix Commands**

For most systems, try this sequence:

```bash
# 1. Install system dependencies (macOS)
brew install libomp

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Force reinstall critical packages
pip uninstall xgboost scikit-learn -y
pip install scikit-learn==1.3.2 xgboost==2.0.3 --no-cache-dir

# 4. Test
python test_optimized_model.py

# 5. Run app
streamlit run src/frontend/app.py
```

---

## 🔄 **Alternative: Use Automated Setup**

If manual fixes don't work, use the automated setup:

```bash
# Run the setup script
./setup.sh        # Linux/macOS
setup.bat         # Windows
```

This will:

- Detect your OS
- Install required system dependencies
- Set up virtual environment
- Install all Python packages
- Test the installation

---

## 📊 **Verification Checklist**

After applying fixes, verify:

- [ ] XGBoost imports without errors: `python -c "import xgboost; print(xgboost.__version__)"`
- [ ] Scikit-learn version correct: `python -c "import sklearn; print(sklearn.__version__)"`
- [ ] Model files exist: `ls models/aml_risk_model_optimized.pkl`
- [ ] Virtual environment active: `which python` points to `.venv`
- [ ] Test script passes: `python test_optimized_model.py`
- [ ] Streamlit shows optimized model: Look for "✅ Loaded XGBoost (Weighted) model"

---

## 🆘 **Still Having Issues?**

If the model still doesn't load:

1. **Share diagnostic output:**

   ```bash
   python diagnose_model.py > diagnostic_output.txt
   ```

2. **Check exact error messages** in the Streamlit console

3. **Verify model file integrity:**

   ```bash
   # Check file size (should be several MB)
   ls -lh models/aml_risk_model_optimized.pkl
   ```

4. **Try the fallback model:**
   - The system should automatically fall back to rule-based scoring
   - While not as accurate, it will still provide risk scores

---

## 💡 **Prevention for Future Deployments**

1. **Use Docker** for consistent environments
2. **Pin exact dependency versions** (already done in requirements.txt)
3. **Document system requirements** clearly
4. **Use the automated setup scripts** for new machines
5. **Test on similar environments** before deployment

The optimized model should work on any system with proper dependencies installed!
