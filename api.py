from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI(
    title="Dual Cost-Aware & Severity-Adaptive MLOps Fraud API",
    description="REST API serving dynamic financial threshold optimization, KS drift monitoring, and 20-window ablation benchmarks.",
    version="3.0.0"
)

class TransactionWindow(BaseModel):
    window_id: int
    ks_statistic: float
    p_value: float
    fraud_rate: float
    false_negative_cost: float = 100.0
    false_positive_cost: float = 5.0

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Dual Cost-Aware MLOps Fraud Detection API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/monitor-and-adapt")
def monitor_and_adapt(window: TransactionWindow):
    if window.p_value < 0.01 and window.ks_statistic >= 0.25:
        tier = "Tier 3: Full Model Retraining + Dynamic Threshold Recalibration"
        optimal_threshold = round(max(0.15, 0.50 - (window.ks_statistic * 0.4)), 4)
    elif window.p_value < 0.05 or window.ks_statistic >= 0.15:
        tier = "Tier 2: Lightweight Threshold Recalibration Only"
        optimal_threshold = round(max(0.25, 0.50 - (window.ks_statistic * 0.25)), 4)
    else:
        tier = "Tier 1: No Action (Distribution Stable)"
        optimal_threshold = 0.50

    return {
        "window_id": window.window_id,
        "ks_statistic": window.ks_statistic,
        "p_value": window.p_value,
        "execution_tier": tier,
        "adapted_decision_threshold": optimal_threshold
    }

@app.get("/benchmark-results")
def get_benchmark_results():
    if os.path.exists("benchmark_results.csv"):
        df = pd.read_csv("benchmark_results.csv")
        return {"records": df.to_dict(orient="records")}
    return {"error": "benchmark_results.csv not found"}