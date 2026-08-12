"""
Fraud Analytics & Risk Engine Streamlit Application.
Provides real-time scoring, financial metrics, and operational triage workflows.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Fraud Analytics & Risk Engine",
    page_icon="💳",
    layout="wide"
)


# =============================================================================
# 1. DATA ENGINEERING PIPELINE METHODS
# =============================================================================

@st.cache_data
def load_and_preprocess_data():
    """
    Load 'creditcard.csv' into RAM to prevent dashboard lag.
    Falls back to synthetic data if the dataset is not present.
    """
    if os.path.exists("creditcard.csv"):
        data_frame = pd.read_csv("creditcard.csv")
    else:
        np.random.seed(42)
        n_samples = 5000
        n_fraud = int(n_samples * 0.0017)

        time_vals = np.sort(np.random.uniform(0, 172800, n_samples))
        amounts = np.random.exponential(scale=88, size=n_samples)
        v_features = {f"V{i}": np.random.normal(0, 1, n_samples) for i in range(1, 29)}

        classes = np.zeros(n_samples, dtype=int)
        fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
        classes[fraud_indices] = 1

        for idx in fraud_indices:
            v_features["V14"][idx] -= 4.0
            v_features["V17"][idx] -= 3.5
            amounts[idx] *= 3.5

        data = {"Time": time_vals, **v_features, "Amount": amounts, "Class": classes}
        data_frame = pd.DataFrame(data)

    return data_frame


@st.cache_resource
def train_imbalance_resolution_model(_df):
    """
    Train a balanced Random Forest classifier and predict fractional risk scores.
    """
    # pylint: disable=invalid-name
    X = _df.drop(columns=["Class"])
    y = _df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    processed_df = _df.copy()
    processed_df["Risk_Score"] = model.predict_proba(X)[:, 1]

    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    return processed_df, matrix


# Load and score dataset
raw_df = load_and_preprocess_data()
df_scored, test_cm = train_imbalance_resolution_model(raw_df)

# Stream Simulation Method: Filter chronologically by 'Time' column
st.sidebar.title("Pipeline Controls")
st.sidebar.subheader("Stream Simulation")

min_time, max_time = int(df_scored["Time"].min()), int(df_scored["Time"].max())
time_window = st.sidebar.slider(
    "Chronological Window (Time Elapsed in Seconds):",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time)
)

risk_threshold = st.sidebar.slider(
    "Operational Risk Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

# Apply chronological stream slice
filtered_df = df_scored[
    (df_scored["Time"] >= time_window[0]) & (df_scored["Time"] <= time_window[1])
].copy()

# Business Metric Synthesis Method
gross_fraud_losses = filtered_df[filtered_df["Class"] == 1]["Amount"].sum()
flagged_mask = filtered_df["Risk_Score"] >= risk_threshold
prevented_fraud_value = filtered_df[flagged_mask & (filtered_df["Class"] == 1)]["Amount"].sum()

tn = len(filtered_df[(~flagged_mask) & (filtered_df["Class"] == 0)])
fp = len(filtered_df[flagged_mask & (filtered_df["Class"] == 0)])
false_positive_ratio = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

# =============================================================================
# 2. DATA VISUALIZATION & INTERFACE METHODS
# =============================================================================

st.title("💳 Credit Card Fraud Analytics & Triage Platform")
st.caption("Dockerized Streamlit Architecture for Financial Anomaly Detection")

# Financial KPI Matrix
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Gross Fraud Losses", f"${gross_fraud_losses:,.2f}")
kpi2.metric("Prevented Fraud Value", f"${prevented_fraud_value:,.2f}")
kpi3.metric("False Positive Ratio", f"{false_positive_ratio:.2%}")
kpi4.metric("Flagged / Total", f"{flagged_mask.sum():,} / {len(filtered_df):,}")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

# Confusion Matrix Heatmap Method
with col_left:
    st.subheader("Model Health Matrix (Confusion Matrix)")
    cm_labels = ["Legitimate (0)", "Fraud (1)"]
    fig_cm = px.imshow(
        test_cm,
        x=cm_labels,
        y=cm_labels,
        text_auto=True,
        color_continuous_scale="Reds",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        aspect="auto"
    )
    fig_cm.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        filtered_df,
        x="Risk_Score",
        color="Class",
        nbins=40,
        barmode="overlay",
        color_discrete_map={0: "#2b5c8f", 1: "#d9534f"},
        labels={"Risk_Score": "Fractional Risk Score", "Class": "Actual Class"}
    )
    fig_hist.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# Filter only actual fraud cases and compute cumulative sum
fraud_df = filtered_df[filtered_df["Class"] == 1].sort_values("Time")

if not fraud_df.empty:
    fraud_df["Cumulative_Loss"] = fraud_df["Amount"].cumsum()

    fig_loss = px.line(
        fraud_df,
        x="Time",
        y="Cumulative_Loss",
        title="Cumulative Gross Fraud Losses Over Time",
        labels={"Time": "Elapsed Time (Seconds)", "Cumulative_Loss": "Gross Losses ($)"},
        markers=True
    )
    fig_loss.update_traces(line_color="#d9534f")
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("No gross fraud losses detected in the current window.")
    
#Filter the flagged Alerts vs Time
st.markdown("---")
st.subheader("Flagged Alerts & Volume Over Time")

# Add a boolean flag column based on risk threshold
threshold_df = filtered_df.copy()
threshold_df["Status"] = threshold_df["Risk_Score"].apply(
    lambda x: "Flagged" if x >= risk_threshold else "Passed"
)

# Bin time into hourly or 10-minute intervals for visualization
threshold_df["Time_Bin"] = (threshold_df["Time"] // 3600).astype(int)

fig_flagged = px.histogram(
    threshold_df,
    x="Time_Bin",
    color="Status",
    title="Transaction Volume & Alert Distribution Over Time",
    labels={"Time_Bin": "Elapsed Time (Hours)", "count": "Transaction Count"},
    color_discrete_map={"Flagged": "#d9534f", "Passed": "#2b5c8f"},
    barmode="stack"
)
fig_flagged.update_layout(margin=dict(l=20, r=20, t=40, b=20))

st.plotly_chart(fig_flagged, use_container_width=True)

# Semantic Alert Color Coding & Actionable Webhook Interface
st.subheader("Operational Anomaly Stream & Human-in-the-Loop Webhook Interface")

high_risk_df = filtered_df[filtered_df["Risk_Score"] >= risk_threshold].copy()

if high_risk_df.empty:
    st.info("No anomalous transactions detected in this chronological window.")
else:
    display_cols = ["Time", "Amount", "Risk_Score", "Class", "V14", "V17"]
    table_data = high_risk_df[display_cols].reset_index(drop=True)

    if "triage_records" not in st.session_state:
        st.session_state.triage_records = {}

    table_data["Webhook Action"] = table_data.index.map(
        lambda idx: st.session_state.triage_records.get(idx, "Pending Review")
    )

    def apply_soft_alert(val):
        """Highlight risk scores exceeding the active threshold."""
        if val >= risk_threshold:
            return "background-color: rgba(217, 83, 79, 0.25); color: #900;"
        return ""

    styled_table = table_data.style.map(
        apply_soft_alert, subset=["Risk_Score"]
    ).format({
        "Amount": "${:,.2f}",
        "Risk_Score": "{:.4f}"
    })

    st.dataframe(styled_table, use_container_width=True)

    # Actionable Webhook Interface Method
    st.markdown("**Actionable Decision Dispatcher**")
    act_col1, act_col2, act_col3 = st.columns([2, 2, 2])

    with act_col1:
        target_row = st.number_input(
            "Select Table Row Index:",
            min_value=0,
            max_value=len(table_data) - 1,
            step=1
        )
    with act_col2:
        action_type = st.selectbox(
            "Set Override Status:",
            ["Approved", "Confirmed Fraud"]
        )
    with act_col3:
        st.write("")
        st.write("")
        if st.button("Dispatch Decision Webhook"):
            st.session_state.triage_records[target_row] = action_type
            st.success(f"Row {target_row} updated to '{action_type}' and dispatched.")
            st.rerun()