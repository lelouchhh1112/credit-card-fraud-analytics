# Real-Time Fraud Analytics & Triage Engine

A dockerized, interactive machine learning dashboard designed to process imbalanced credit card transaction feeds, evaluate real-time fraud risk scores, and manage analyst triage workflows.

---

## Technical Overview

* **Data Ingestion & Caching**: Ingests Kaggle Credit Card Fraud data with an automated synthetic fallback generator and in-memory caching via `@st.cache_data`.
* **ML Machine Learning Pipeline**: Trains a balanced Random Forest model to calculate real-time continuous risk scores ($0.00$–$1.00$) over high-dimensional transaction features ($V1$–$V28$).
* **Interactive Triage Workflow**: Enables dynamic temporal filtering, threshold adjustments, financial loss calculation, and analyst decision overrides.

---

## Repository Structure

```text
.
├── app.py              # Main application entry point & Streamlit dashboard
├── creditcard.csv       # Dataset storage location (git-ignored)
├── Dockerfile          # Container image blueprint
├── docker-compose.yml  # Multi-container orchestration and volume mounting
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
