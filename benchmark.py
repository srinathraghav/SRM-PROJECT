import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp

# 1. Generate Reproducible 25,000-Transaction Stream
np.random.seed(42)
n_samples, n_fraud = 25000, 45
n_normal = n_samples - n_fraud

features = np.vstack([
    np.random.normal(0, 1, size=(n_normal, 28)),
    np.random.normal(1.5, 2.0, size=(n_fraud, 28))
])
df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
df["Class"] = [0] * n_normal + [1] * n_fraud
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

X_base, X_stream, y_base, y_stream = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

scaler = StandardScaler()
X_base_s = pd.DataFrame(scaler.fit_transform(X_base), columns=X.columns)
X_stream_s = pd.DataFrame(scaler.transform(X_stream), columns=X.columns).reset_index(drop=True)
y_stream = y_stream.reset_index(drop=True)

# 2. Cost Parameters (INR)
COST_FN = 25000.0
COST_FP = 500.0
BAYES_THRESHOLD = COST_FP / (COST_FP + COST_FN)  # 0.0196

# 20 Sequential Production Batches (500 transactions per window)
drift_schedule = [
    0.0, 0.0, 0.05, 0.30, 0.35,   # Windows 1-5: Normal -> Minor Drift
    0.40, 0.10, 0.0, 1.40, 1.60,  # Windows 6-10: Minor -> Severe Drift Shock
    0.35, 0.10, 0.0, 0.0, 0.32,   # Windows 11-15: Recovery -> Minor Drift
    1.50, 0.40, 0.10, 0.0, 0.0    # Windows 16-20: Second Shock -> Stabilization
]

def evaluate_batch(model, X_b, y_b, threshold, compute_cost):
    probs = model.predict_proba(X_b)[:, 1]
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_b, preds, labels=[0, 1]).ravel()
    fin_loss = (fn * COST_FN) + (fp * COST_FP)
    return compute_cost, fin_loss, compute_cost + fin_loss, fn, fp

def init_base_model():
    m = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    m.fit(X_base_s, y_base)
    return m

# Pre-compute balanced class weights dictionary for partial_fit
CLASSES = np.array([0, 1])
WEIGHTS = compute_class_weight("balanced", classes=CLASSES, y=y_base)
CW_DICT = {0: WEIGHTS[0], 1: WEIGHTS[1]}

# Initialize Models for all 4 Strategies
m_static = init_base_model()
m_periodic = init_base_model()
m_standard = init_base_model()
m_proposed = init_base_model()

results = {
    "1. Static Baseline (No Retrain)": {"compute": 0.0, "financial": 0.0, "fn": 0, "fp": 0, "retrains": 0},
    "2. Naive Periodic (Every 5 Batches)": {"compute": 0.0, "financial": 0.0, "fn": 0, "fp": 0, "retrains": 0},
    "3. Standard Drift-Triggered (All-Tier-2)": {"compute": 0.0, "financial": 0.0, "fn": 0, "fp": 0, "retrains": 0},
    "4. Proposed Dual Cost-Aware Adaptive": {"compute": 0.0, "financial": 0.0, "fn": 0, "fp": 0, "retrains": 0},
}

batch_size = len(X_stream_s) // len(drift_schedule)

for w_idx, d_factor in enumerate(drift_schedule):
    start, end = w_idx * batch_size, (w_idx + 1) * batch_size
    X_b = X_stream_s.iloc[start:end].copy()
    y_b = y_stream.iloc[start:end]

    if d_factor > 0:
        for col in ["V1", "V2", "V3"]:
            X_b[col] += np.random.normal(d_factor * 1.5, d_factor, size=len(X_b))

    ks_drift = float(np.mean([ks_2samp(X_base_s[c].values, X_b[c].values)[0] for c in ["V1", "V2", "V3"]]))

    # --- Strategy 1: Static ---
    c, f, _, fn, fp = evaluate_batch(m_static, X_b, y_b, 0.50, 50.0)
    results["1. Static Baseline (No Retrain)"]["compute"] += c
    results["1. Static Baseline (No Retrain)"]["financial"] += f
    results["1. Static Baseline (No Retrain)"]["fn"] += fn
    results["1. Static Baseline (No Retrain)"]["fp"] += fp

    # --- Strategy 2: Periodic (Every 5 windows) ---
    if (w_idx + 1) % 5 == 0:
        m_periodic.fit(pd.concat([X_base_s, X_b]), pd.concat([y_base, y_b]))
        comp_2 = 1450.0
        results["2. Naive Periodic (Every 5 Batches)"]["retrains"] += 1
    else:
        comp_2 = 50.0
    c, f, _, fn, fp = evaluate_batch(m_periodic, X_b, y_b, 0.50, comp_2)
    results["2. Naive Periodic (Every 5 Batches)"]["compute"] += c
    results["2. Naive Periodic (Every 5 Batches)"]["financial"] += f
    results["2. Naive Periodic (Every 5 Batches)"]["fn"] += fn
    results["2. Naive Periodic (Every 5 Batches)"]["fp"] += fp

    # --- Strategy 3: Standard Drift MLOps (Any drift >= 0.15 -> Full Tier 2, 0.50 threshold) ---
    if ks_drift >= 0.15:
        m_standard.fit(pd.concat([X_base_s, X_b]), pd.concat([y_base, y_b]))
        comp_3 = 1450.0
        results["3. Standard Drift-Triggered (All-Tier-2)"]["retrains"] += 1
    else:
        comp_3 = 50.0
    c, f, _, fn, fp = evaluate_batch(m_standard, X_b, y_b, 0.50, comp_3)
    results["3. Standard Drift-Triggered (All-Tier-2)"]["compute"] += c
    results["3. Standard Drift-Triggered (All-Tier-2)"]["financial"] += f
    results["3. Standard Drift-Triggered (All-Tier-2)"]["fn"] += fn
    results["3. Standard Drift-Triggered (All-Tier-2)"]["fp"] += fp

    # --- Strategy 4: Proposed Severity-Adaptive + Bayes Cost-Optimal Threshold ---
    if ks_drift < 0.15:
        comp_4 = 50.0
    elif ks_drift < 0.45:
        sgd = SGDClassifier(loss="log_loss", class_weight=CW_DICT, random_state=42)
        sgd.partial_fit(X_base_s, y_base, classes=CLASSES)
        sgd.partial_fit(X_b, y_b)
        m_proposed = sgd
        comp_4 = 320.0
        results["4. Proposed Dual Cost-Aware Adaptive"]["retrains"] += 1
    else:
        m_proposed = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        m_proposed.fit(pd.concat([X_base_s, X_b]), pd.concat([y_base, y_b]))
        comp_4 = 1450.0
        results["4. Proposed Dual Cost-Aware Adaptive"]["retrains"] += 1

    c, f, _, fn, fp = evaluate_batch(m_proposed, X_b, y_b, BAYES_THRESHOLD, comp_4)
    results["4. Proposed Dual Cost-Aware Adaptive"]["compute"] += c
    results["4. Proposed Dual Cost-Aware Adaptive"]["financial"] += f
    results["4. Proposed Dual Cost-Aware Adaptive"]["fn"] += fn
    results["4. Proposed Dual Cost-Aware Adaptive"]["fp"] += fp

# 3. Format and Print Summary Table
rows = []
for name, vals in results.items():
    total_dual = vals["compute"] + vals["financial"]
    rows.append({
        "Strategy": name,
        "Retrains": vals["retrains"],
        "Missed Fraud (FN)": vals["fn"],
        "False Positives (FP)": vals["fp"],
        "Compute Cost (INR)": f"Rs. {vals['compute']:,.2f}",
        "Financial Loss (INR)": f"Rs. {vals['financial']:,.2f}",
        "Total Dual Cost (INR)": f"Rs. {total_dual:,.2f}",
    })

summary_df = pd.DataFrame(rows)
print("\n=== 20-WINDOW COMPARATIVE ABLATION BENCHMARK ===")
print(summary_df.to_string(index=False))
summary_df.to_csv("benchmark_results.csv", index=False)
print("\nSaved results to benchmark_results.csv")