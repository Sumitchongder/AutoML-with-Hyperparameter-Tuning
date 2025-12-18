# 🚀 HYPERML: ACCELERATING AI WITH ADVANCED AUTOMATION

[![GitHub Stars](https://img.shields.io/github/stars/Sumitchongder/AutoML-with-Hyperparameter-Tuning?style=social)](https://github.com/Sumitchongder/AutoML-with-Hyperparameter-Tuning)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-brightgreen.svg)](https://streamlit.io/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-orange.svg)](https://pycaret.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### **A No-Code Machine Learning Platform Built with PyCaret + Streamlit**

This project is a **complete, production-ready AutoML web app**.
Users can upload a dataset, explore data, train ML models, compare them, visualize results, and download a final model all through a simple GUI.

---

## 📜 Copyright Registration (Government of India)

<p align="center">
   <img width="600" height="1200" alt="Image" src="https://github.com/user-attachments/assets/47475ac4-60d3-4e47-902b-161672c87a96" />
</p>

---

## 🌟 Features

### 📁 **1. Upload Any CSV Dataset**

* Automatic data type detection
* Missing values summary
* Data preview + statistics
* Clear UI workflow (Data → Training → Visualization → Prediction)

---

### 🤖 **2. AutoML Engine (Powered by PyCaret)**

* Automatic preprocessing
* Cross-validation
* Compare multiple ML models
* Auto-select best model
* Manual model selection supported

---

### 📊 **3. Visualizations**

* Confusion Matrix
* ROC / PR Curve
* Residuals
* Feature Importance
* Error plots
* All charts rendered safely without caching errors

---

### 📦 **4. Finalize Model**

* Save the model safely
* Prevents filename duplication
* Download `.pkl` model file
* Load model for predictions

---

### 🔮 **5. Make Predictions**

* Upload CSV for batch inference
* Manual form for single prediction
* Download prediction results as CSV

---

### 🔁 **6. Deployment-Safe Design**

* Full Streamlit session state handling
* Bug-free PyCaret setup
* No excessive memory usage
* Safe chart rendering
* No threading errors

---

# 🏗️ Project Structure

```
AutoML-App/
│
├── streamlit_app_automl.py   # Main Streamlit application
├── requirements.txt          # All dependencies
├── models/                   # Auto-generated saved models
├── temp/                     # Temporary experiment/charts
├── assets/                   # Logos, icons (optional)
└── README.md                 # Documentation (this file)
```

---

# 🧰 **Installation Guide (Beginner Friendly)**

This section is step-by-step with **zero assumptions**.

---

## ✔️ Step 1: Install Python 3.10 or 3.11

Download and install Python from:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation **check the box**:

```
☑ Add Python to PATH
```

---

## ✔️ Step 2: Download or Clone the Repository

### **Option A — Clone using Git**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **Option B — Download ZIP**

1. Click **Code → Download ZIP**
2. Extract it
3. Open the extracted folder

---

# 🧪 Step 3: Create a Virtual Environment

### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see:

```
(venv) C:\YourProject>
```

---

# 📦 Step 4: Install All Requirements

```bash
pip install -r requirements.txt
```

This installs:

* Streamlit
* PyCaret
* Scikit-learn
* Pandas
* Plotly
* Matplotlib

Everything required to run the app.

---

# ▶️ Step 5: Run the AutoML App

```bash
streamlit run streamlit_app_automl.py
```

Now your browser will automatically open the app at:

👉 [http://localhost:8501](http://localhost:8501)

---

# 🔧 Troubleshooting (Beginner Friendly)

### ❗ “streamlit: command not found”

Your virtual environment is not activated.
Run:

➡ **Windows**

```bash
venv\Scripts\activate
```

➡ **Mac/Linux**

```bash
source venv/bin/activate
```

---

### ❗ PyCaret setup errors

Restart the app:

```bash
streamlit run streamlit_app_automl.py
```

---

### ❗ Model not saving

Make sure the repo has:

```
models/
temp/
```

If missing, create them manually.

---

# 🌐 Deployment

## 🚀 Deploy to Streamlit Cloud

1. Push repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repo
4. Add this in "Python version":

   ```
   3.10
   ```
5. Deploy ✔️

No extra config required.

---

# 🧭 Future Improvements

* SHAP explainability
* Auto PDF report generation
* Model monitoring dashboard
* Multi-page UI
* Cloud model registry

---

# 👨‍💻 Author

**Sumit Chongder**
Machine Learning Engineer | AutoML Systems | Quantum & AI Research

🔗 LinkedIn: https://www.linkedin.com/in/sumit-chongder/

---

# 🎉 Support the Project

If this project helped you, please ⭐ **star the GitHub repo** — it motivates further development!




