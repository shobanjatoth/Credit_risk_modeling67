# 🚀 Credit Risk Modelling System with MLOps

## 📌 Project Overview

This project is an end-to-end Credit Risk Modelling System designed to predict the likelihood of loan default using machine learning and production-grade MLOps practices.

The solution covers the complete machine learning lifecycle including:

* Data preprocessing
* Feature engineering
* Model training and evaluation
* Experiment tracking
* Model versioning
* API deployment
* Interactive business dashboard
* Containerization
* Automated testing
* CI/CD automation

The primary objective is to help financial institutions assess credit risk efficiently and make data-driven lending decisions.

---

## 🎯 Business Problem

Loan defaults pose significant financial risks to banks and lending institutions. Traditional manual credit assessment processes are often time-consuming and inconsistent.

This system predicts whether a loan applicant is likely to default, enabling:

* Faster loan approval processes
* Reduced financial losses
* Improved risk management
* Better customer segmentation

---

## 🏗️ System Architecture

```text
Raw Data
    │
    ▼
Data Validation
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Model Training (XGBoost)
    │
    ▼
MLflow Tracking & Versioning
    │
    ▼
FastAPI Deployment
    │
    ▼
Streamlit Dashboard
    │
    ▼
Docker Containerization
    │
    ▼
CI/CD using GitHub Actions
```

---

## 🛠️ Tech Stack

### Programming Language
* Python

### Machine Learning
* XGBoost
* Scikit-Learn
* Pandas
* NumPy

### Experiment Tracking
* MLflow

### API Development
* FastAPI
* Uvicorn

### Frontend
* Streamlit

### DevOps & MLOps
* Docker
* GitHub Actions
* Pytest

---

## 📂 Project Structure

```bash
Credit-Risk-Modelling/
│
├── artifacts/
├── config/
├── notebooks/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── utils/
│   └── logger/
│
├── tests/
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/
└── README.md
```

---

## ⚙️ Features

### Data Processing
* Missing value handling
* Outlier treatment
* Feature encoding
* Feature scaling

### Model Development
* XGBoost classifier
* Hyperparameter tuning
* Cross-validation
* Model evaluation

### MLOps
* MLflow experiment tracking
* Model version management
* Automated unit testing
* Dockerized deployment
* CI/CD pipeline

### Deployment
* FastAPI prediction service
* Streamlit business dashboard
* REST API documentation

---

## 📊 Model Performance

### Evaluation Metrics

![Evaluation Metrics](images/evaluation_metrics.png)

### Key Metrics

|       Class      |  Precision |     Recall |   F1-Score |    Support |
| :--------------: | ---------: | ---------: | ---------: | ---------: |
|         0        |     0.7356 |     0.8872 |     0.8043 |      1,135 |
|         1        |     0.8938 |     0.8389 |     0.8655 |      6,373 |
|         2        |     0.4456 |     0.4316 |     0.4385 |      1,536 |
|         3        |     0.7142 |     0.8350 |     0.7699 |      1,218 |
|   **Macro Avg**  | **0.6973** | **0.7482** | **0.7195** | **10,262** |
| **Weighted Avg** | **0.7879** | **0.7828** | **0.7834** | **10,262** |

---

## 🔍 MLflow Experiment Tracking

MLflow was used for:

* Experiment tracking
* Hyperparameter logging
* Metric monitoring
* Model versioning

### MLflow Dashboard
![MLflow Experiments](images/mlflow_experiments.png)

### Logged Parameters
![MLflow Parameters](<img width="877" height="352" alt="Screenshot 2026-06-18 150108" src="https://github.com/user-attachments/assets/2dba355e-1588-456a-a1e2-c02fe24f0862" />
/mlflow_parameters.png)

### Registered Models
![MLflow Models](<img width="1007" height="218" alt="Screenshot 2026-06-18 150222" src="https://github.com/user-attachments/assets/456916e3-ef0b-416f-bb3c-76273567144e" />
)

---

## 🌐 FastAPI Deployment

The trained model is exposed through FastAPI endpoints.

### API Documentation
![FastAPI Docs](images/fastapi_docs.png)

### Prediction Endpoint

```http
POST /predict
```

### Sample Request

```json
{
  "Age": 35,
  "Income": 50000,
  "LoanAmount": 200000
}
```

### Sample Response

```json
{
  "prediction": 0,
  "risk_category": "Low Risk"
}
```

---

## 📱 Streamlit Dashboard

A business-friendly dashboard was developed using Streamlit for real-time predictions.

### User Interface
![Streamlit UI](images/streamlit_ui.png)

### Dashboard Features
* Easy data input
* Real-time predictions
* Risk categorization
* Interactive user experience

---

## 🐳 Docker Containerization

The entire application is containerized using Docker.

### Build Docker Image

```bash
docker build -t credit-risk-app .
```

### Run Container

```bash
docker run -p 8000:8000 credit-risk-app
```

---

## 🔄 CI/CD Pipeline

GitHub Actions automates:

* Code validation
* Unit testing
* Docker build verification
* Deployment workflow

### Workflow
![GitHub Actions](images/github_actions.png)

---

## 🧪 Testing

Automated unit tests ensure reliability and maintainability.

Run tests:

```bash
pytest
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/credit-risk-modelling.git
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI

```bash
uvicorn main:app --reload
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

* Model monitoring and drift detection
* Cloud deployment (AWS/Azure/GCP)
* Feature Store integration
* Real-time batch predictions
* Explainable AI using SHAP

---

## 👨‍💻 Author

**Jatoth Shoban Babu**

B.Tech (Computer Science - Data Science)

Skills:

* Machine Learning
* Deep Learning
* MLOps
* Python
* SQL
* FastAPI
* Streamlit
* Docker
* MLflow

---

## 🏁 Key Takeaways

This project demonstrates:

* End-to-End Machine Learning Pipeline
* Production-grade MLOps Practices
* Experiment Tracking with MLflow
* API Deployment using FastAPI
* Interactive Dashboard Development
* Docker Containerization
* CI/CD Automation with GitHub Actions

A complete industry-level implementation of Credit Risk Modeling from development to deployment.
