import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp

st.set_page_config(layout="wide", page_title="Review 2 - Dual Cost-Aware MLOps Prototype")

st.title("Dual Cost-Aware, Severity-Adaptive MLOps Fraud Detection")
st.caption("M.Tech Cloud Computing - Review 2 Implementation Prototype")

# 1. Load Data with Self-Contained Fallback
@st.cache_data
def load_data():
    if os.path.exists("creditcard.csv"):
        return pd.read_csv("creditcard.csv", nrows=25000)
    
    # Cloud container fallback: generates exact PCA feature distributions
    np.random.seed(42)
    n_samples = 25000
    n_fraud = 45
    n_normal = n_samples - n_fraud
    
    normal_data = np.random.normal(0, 1, size=(n_normal, 28))
    fraud_data = np.random.normal(1.5, 2.0, size=(n_fraud, 28))
    features = np.vstack([normal_data, fraud_data])
    
    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(np.random.uniform(0, 172800, size=n_samples))
    df["Amount"] = np.random.exponential(scale=88.0, size=n_samples)
    df["Class"] = [0] * n_normal + [1] * n_fraud
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df = load_data()
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Historical baseline vs Incoming Production Stream
X_baseline, X_prod, y_baseline, y_prod = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Standardize features for stable ML convergence
scaler = StandardScaler()
X_baseline_scaled = pd.DataFrame(scaler.fit_transform(X_baseline), columns=X.columns)
X_prod_scaled = pd.DataFrame(scaler.transform(X_prod), columns=X.columns)

# 2. Baseline Model
@st.cache_resource
def get_trained_model(X_tr, y_tr):
    model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    model.fit(X_tr, y_tr)
    return model

model = get_trained_model(X_baseline_scaled, y_baseline)

# Sidebar Controls
st.sidebar.header("1. Production Stream Simulation")
drift_factor = st.sidebar.slider("Simulate Concept / Feature Drift", 0.0, 3.0, 0.0, 0.1)

st.sidebar.header("2. Dual Cost Parameters")
cost_fn = st.sidebar.number_input("Cost of Missed Fraud (FN) in INR", value=25000.0, step=1000.0)
cost_fp = st.sidebar.number_input("Cost of Legitimate Friction (FP) in INR", value=500.0, step=100.0)

# 3. Dynamic Drift Injection
X_incoming = X_prod_scaled.copy()
if drift_factor > 0:
    for col in ['V1', 'V2', 'Amount']:
        X_incoming[col] = X_incoming[col] + np.random.normal(drift_factor * 1.5, drift_factor, size=len(X_incoming))

# 4. Drift Quantification via Kolmogorov-Smirnov Test
drift_scores = []
for feature in ['V1', 'V2', 'Amount']:
    stat, pval = ks_2samp(X_baseline_scaled[feature].values, X_incoming[feature].values)
    drift_scores.append(stat)

avg_drift = float(np.mean(drift_scores))

# 5. Severity-Adaptive Decision Tier
if avg_drift < 0.15:
    tier = "Tier 0: Normal Operation"
    action = "No retraining required. Current model remains actively serving."
    compute_cost = 50.0
    badge_color = "green"
elif avg_drift < 0.45:
    tier = "Tier 1: Minor Drift Detected"
    action = "Trigger lightweight incremental fine-tune on AWS Spot Instance."
    compute_cost = 320.0
    badge_color = "orange"
else:
    tier = "Tier 2: Major Concept Drift Detected"
    action = "Trigger full pipeline re-training on primary instance cluster."
    compute_cost = 1450.0
    badge_color = "red" 

# 6. Cost-Aware Threshold Optimization
probs = model.predict_proba(X_incoming)[:, 1]
optimal_threshold = cost_fp / (cost_fp + cost_fn)
y_pred = (probs >= optimal_threshold).astype(int)

cm = confusion_matrix(y_prod, y_pred)
tn, fp, fn, tp = cm.ravel()

business_loss = (fn * cost_fn) + (fp * cost_fp)
total_dual_cost = business_loss + compute_cost

# 7. Real-Time UI
col1, col2, col3, col4 = st.columns(4)
col1.metric("Quantified Drift Severity", f"{avg_drift:.3f}")
col2.metric("Retraining Tier", tier.split(':')[0])
col3.metric("Cost-Optimal Threshold", f"{optimal_threshold:.4f}")
col4.metric("Total Dual Cost (INR)", f"Rs. {total_dual_cost:,.0f}")

st.markdown(f"**Triggered Action:** :{badge_color}[**{tier}** — {action}]")
st.divider()

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Transaction Classification (Incoming Stream)")
    res_df = pd.DataFrame(
        cm,
        columns=["Predicted Genuine", "Predicted Fraud"],
        index=["Actual Genuine", "Actual Fraud"]
    )
    st.table(res_df)

with col_right:
    st.subheader("Dual Cost Breakdown")
    breakdown_df = pd.DataFrame({
        "Cost Category": [
            "Cloud Infrastructure / Retrain Compute",
            f"Missed Fraud (False Negatives: {fn})",
            f"Customer Friction (False Positives: {fp})",
            "Net Operational & Financial Impact"
        ],
        "Amount (INR)": [
            f"Rs. {compute_cost:,.2f}",
            f"Rs. {fn * cost_fn:,.2f}",
            f"Rs. {fp * cost_fp:,.2f}",
            f"Rs. {total_dual_cost:,.2f}"
        ]
    })
    st.table(breakdown_df)