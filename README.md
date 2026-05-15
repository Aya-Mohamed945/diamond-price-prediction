# 💎 Diamond Price Prediction System

An end-to-end Machine Learning project for predicting diamond prices using multiple regression algorithms including Linear Regression, K-Nearest Neighbors (KNN), and Support Vector Regression (SVR).

The project follows a professional modular architecture with configurable pipelines, feature engineering, hyperparameter tuning, visualization utilities, and experiment-ready configurations.

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Machine Learning Models](#-machine-learning-models)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration System](#-configuration-system)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Visualizations](#-visualizations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

# 🎯 Project Overview

This project predicts diamond prices based on several physical and categorical characteristics such as:

- Carat
- Cut
- Color
- Clarity
- Dimensions (x, y, z)
- Table
- Depth

The system was designed using a scalable and production-style structure commonly used in real-world Machine Learning projects.

The project includes:
- Data preprocessing pipeline
- Feature engineering
- Multiple ML models
- Hyperparameter tuning
- Metrics evaluation
- Visualization utilities
- Config-driven architecture

---

# 💼 Business Problem

Diamond pricing depends on multiple factors and can vary significantly depending on quality and dimensions.

The objective of this project is to build a predictive system capable of estimating diamond prices accurately using Machine Learning techniques.

Potential business applications include:
- Jewelry pricing systems
- E-commerce platforms
- Diamond valuation tools
- Automated pricing engines

---

# 📊 Dataset

Dataset used:

```text
Diamonds - Regression.csv
```

Main Features:

| Feature | Description |
|---|---|
| carat | Weight of the diamond |
| cut | Diamond cut quality |
| color | Diamond color grade |
| clarity | Diamond clarity grade |
| depth | Total depth percentage |
| table | Width of top relative to widest point |
| x,y,z | Physical dimensions |
| price | Target variable |

---

# 🏗️ Project Architecture

The project follows a modular architecture inspired by industry-standard ML pipelines.

Key architectural concepts:
- Separation of concerns
- Config-driven workflow
- Reusable components
- Scalable structure
- Clean code principles

---

# ✨ Features

## ✅ Data Processing
- Data loading abstraction
- Missing value inspection
- Duplicate detection
- Label encoding
- Feature scaling

## ✅ Feature Engineering
- Interaction features
- Volume-based calculations
- Depth anomaly detection

## ✅ Machine Learning
- Linear Regression
- KNN Regressor
- Support Vector Regression (SVR)

## ✅ Model Evaluation
- MAE
- MSE
- RMSE
- R² Score

## ✅ Hyperparameter Tuning
- GridSearchCV integration
- Cross-validation support
- Automated parameter optimization

## ✅ Visualization
- Correlation matrix
- Prediction plots
- Residual analysis

## ✅ Configuration Management
- YAML-based configs
- Environment flexibility
- Easy experimentation
- Centralized settings

---

# 🛠️ Tech Stack

## Programming Language
- Python 3.x

## Libraries
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- Joblib

## Tools
- Jupyter Notebook
- Git
- GitHub

---

# 🤖 Machine Learning Models

| Model | Description |
|---|---|
| Linear Regression | Baseline regression model |
| KNN Regressor | Distance-based regression |
| SVR | Support Vector Machine for regression |

---

# 📈 Performance Metrics

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.9057 | \$1224.60 | \$805.27 |
| KNN Regressor | **0.9681** | **\$712.63** | **\$374.34** |
| SVR | 0.5345 | \$2720.31 | \$1268.50 |

## 🏆 Best Model

### K-Nearest Neighbors Regressor (KNN)

- Highest R² Score
- Lowest RMSE
- Best overall prediction accuracy

---

# 📂 Project Structure

```text
diamond-price-prediction/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── config/
│   ├── config.yaml
│   ├── features.yaml
│   ├── preprocessing.yaml
│   └── logging.yaml
│
├── data/
│   └── Diamonds - Regression.csv
│
├── notebooks/
│   └── 01_initial_analysis.ipynb
│
├── src/
│   ├── config_manager.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── linear_regression.py
│   │   ├── knn_model.py
│   │   └── svr_model.py
│   │
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
│
├── models/
│   ├── knn_best_model.pkl
│   ├── svr_best_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── main.py
└── train.py
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/diamond-price-prediction.git
```

## 2️⃣ Navigate to the Project

```bash
cd diamond-price-prediction
```

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## Run Main Pipeline

```bash
python main.py
```

## Run Hyperparameter Tuning

```bash
python train.py
```

---

# ⚙️ Configuration System

The project uses YAML-based configuration files to manage:
- Data settings
- Model parameters
- Feature engineering
- Training options
- Logging
- Visualization settings

Example:

```yaml
models:
  knn:
    n_neighbors: 5
    metric: "euclidean"
```

## Benefits of Config-Based Design

- No hardcoding
- Easier experimentation
- Cleaner architecture
- Better maintainability
- Environment flexibility
- Easier collaboration

---

# 🔍 Hyperparameter Tuning

Implemented using:

```python
GridSearchCV
```

Tuned parameters include:
- KNN neighbors
- Distance metrics
- SVR kernels
- Regularization values
- Epsilon values

---

# 📉 Visualizations

The project generates:
- Correlation heatmaps
- Actual vs Predicted plots
- Residual analysis plots

These visualizations help evaluate:
- Model accuracy
- Error distribution
- Feature relationships

---

# 🚀 Future Improvements

- XGBoost Regressor
- Random Forest Regressor
- CatBoost
- LightGBM
- Deep Learning models
- Docker containerization
- FastAPI deployment
- CI/CD pipeline
- MLflow experiment tracking
- Streamlit dashboard

---

# 🧠 Software Engineering Concepts Applied

- Object-Oriented Programming (OOP)
- Modular Design
- Configuration Management
- Clean Architecture
- Separation of Concerns
- Reusable Components
- Production-style ML Pipeline

---

# 📌 GitHub Best Practices Used

- Meaningful folder structure
- Modular source code
- Environment isolation
- Config-driven architecture
- Git ignore optimization
- Reusable utilities
- Documentation-first development

---

# 👩‍💻 Author

## Aya Mohamed

Machine Learning Engineer | Data Analyst

Passionate about:
- Machine Learning
- Data Science
- MLOps
- Software Engineering

---

# ⭐ If You Like This Project

Please consider:
- Starring the repository
- Forking the project
- Contributing improvements

---

# 📜 License

This project is licensed under the MIT License.