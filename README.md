# Employee Salary Prediction App

A machine learning-powered web application that predicts Indian employee salaries(Monthly) based on various factors such as age, education, experience, job title, and gender, built with Streamlit.

## Live Demo

**Try the app here:** https://salary-app-cmxxzz8v9roapeewap4nxr.streamlit.app/

---

## 🎯 Overview

This project demonstrates an end-to-end machine learning workflow, including:
- Data collection and preprocessing
- Feature engineering and encoding
- Model training and evaluation
- Web application development
- Cloud deployment

The application uses two regression models (Linear Regression and Random Forest) to predict employee salaries with up to 98% accuracy.

---

##  Features

-  **Dual ML Models** - Choose between Linear Regression and Random Forest
-  **Real-time Predictions** - Instant salary estimates based on input
-  **High Accuracy** - Random Forest model achieves 98% R² score
-  **Responsive Design** - Works on desktop and mobile devices
-  **Data Preprocessing** - Automated feature scaling and encoding
-  **Model Caching** - Fast loading using Streamlit's caching mechanism

---

## 📊 Dataset

**Source:** [Kaggle - Salary_Data Dataset](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data)

The dataset contains employee information with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Employee age (18-65 years) |
| Education Level | Categorical | High School, Bachelor's, Master's, PhD |
| Years of Experience | Numerical | Professional experience (0-40 years) |
| Job Title | Categorical | 200+ unique job positions |
| Gender | Categorical | Male, Female, Other |
| Salary | Numerical | Annual salary (Target variable) |

### Data Preprocessing Steps

1. **Data Cleaning**
   - Handled missing values
   - Removed duplicates
   - Outlier detection and treatment

2. **Feature Engineering**
   - Mean encoding for job titles
   - One-hot encoding for gender
   - Label encoding for education levels
   - Standard scaling for numerical features

3. **Train-Test Split**
   - 80% training data
   - 20% testing data

---

## 🤖 Machine Learning Models

### 1. Linear Regression
- Simple baseline model
- Assumes a linear relationship between features and salary
- **Performance:** R² Score = 80%

### 2. Random Forest Regression
- Ensemble learning method
- Handles non-linear relationships
- **Performance:** R² Score = 98%

### Model Training Pipeline

```
Raw Data → Cleaning → Feature Engineering → Scaling → Model Training → Evaluation → Serialization
```

All trained models and preprocessing components are saved as `.pkl` files using `joblib` for efficient loading in the web app.

---

## 🛠️ Tech Stack

**Frontend & Backend:**
- [Streamlit](https://streamlit.io/) - Web application framework

**Machine Learning:**
- [scikit-learn](https://scikit-learn.org/) - Model training and preprocessing
- [joblib](https://joblib.readthedocs.io/) - Model serialization

**Data Processing:**
- [Pandas](https://pandas.pydata.org/) - Data manipulation (used in training)
- [NumPy](https://numpy.org/) - Numerical computations (used in training)

**Deployment:**
- [Streamlit Community Cloud](https://streamlit.io/cloud) - Free cloud hosting

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

## 🚀 Usage

1. **Select Model**: Choose between Linear Regression or Random Forest from the sidebar
2. **Enter Details**: Input employee information:
   - Age (18-65)
   - Education Level
   - Years of Experience (0-40)
   - Job Title
   - Gender
3. **Predict**: Click the "EXECUTE PREDICTION" button
4. **View Results**: See the predicted Monthly salary with confidence metrics according to indian Standards.

