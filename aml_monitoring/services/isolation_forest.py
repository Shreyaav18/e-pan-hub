"""
Isolation Forest Anomaly Detection - REAL SAML-D DATASET
---------------------------------------------------------
Trained on real transaction data from Kaggle SAML-D dataset.
18-dimensional features from actual transactions.

Falls back to synthetic data if no trained model found.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime, timedelta


# ── Load trained model ────────────────────────────────────────────────────

_clf = None

def _get_classifier():
    """Load trained model from disk OR fallback to synthetic"""
    global _clf
    if _clf is None:
        model_path = 'aml_monitoring/models/isolation_forest.pkl'

        if os.path.exists(model_path):
            _clf = joblib.load(model_path)
            print(f"✓ Loaded REAL trained model: {model_path}")
        else:
            print("⚠ No trained model found. Using synthetic fallback.")
            population = _build_synthetic_population(n_samples=500)
            _clf = IsolationForest(
                n_estimators=200,
                max_samples='auto',
                contamination=0.001,
                random_state=42,
                n_jobs=-1
            )
            _clf.fit(population)

    return _clf


# ── Feature engineering (18 dimensions) ───────────────────────────────────

def build_transaction_features(transaction, history=None):
    """
    Build 18-dimensional feature vector for transaction.

    Features:
        f1-f9:   Basic transaction features
        f10:     Currency mismatch (payment != received)
        f11-f17: Historical/velocity features
        f18:     Cross-border flag
    """
    history = history or []
    ts = _parse_time(transaction.get('timestamp'))
    amount = float(transaction.get('amount', 0))

    # Location mapping (FULL NAMES, not ISO codes)
    location_map = {
        'Iran', 'North Korea', 'Syria', 'Myanmar', 'Yemen',
        'Iraq', 'Libya', 'Somalia', 'Pakistan', 'Afghanistan',
        'South Sudan', 'Central African Republic', 'Democratic Republic of Congo',
        'Mali', 'Nicaragua', 'Mexico', 'Turkey', 'Morocco', 'UAE',
        'Tajikistan', 'Uzbekistan'
    }

    sender_loc = str(transaction.get('sender_bank_location', '')).strip()
    receiver_loc = str(transaction.get('receiver_bank_location', '')).strip()
    payment_curr = str(transaction.get('payment_currency', '')).upper()
    received_curr = str(transaction.get('received_currency', '')).upper()
    payment_type = str(transaction.get('payment_type', 'other')).lower()

    # Time features
    hour = ts.hour
    dow = ts.weekday()
    is_weekend = int(dow >= 5)
    is_odd = int(hour >= 23 or hour < 5)

    # Payment type encoding
    type_encoding = {
        'cross-border': 0.9,
        'ach': 0.6,
        'credit card': 0.4,
        'debit card': 0.3,
        'cash': 0.5,
        'cheque': 0.2
    }
    txn_type_encoded = next((v for k, v in type_encoding.items() if k in payment_type), 0.3)

    # Location risks (LOCATION NAMES, not ISO)
    sender_risk = 1.0 if sender_loc in location_map else 0.0
    receiver_risk = 1.0 if receiver_loc in location_map else 0.0

    # Currency mismatch
    currency_mismatch = 1.0 if payment_curr != received_curr else 0.0

    # History-based features
    cutoff_24h = ts - timedelta(hours=24)
    cutoff_7d = ts - timedelta(days=7)

    recent_24h = [t for t in history if _parse_time(t.get('timestamp')) >= cutoff_24h]
    recent_7d = [t for t in history if _parse_time(t.get('timestamp')) >= cutoff_7d]

    txn_count_24h = len(recent_24h)
    total_24h = sum(float(t.get('amount', 0)) for t in recent_24h)
    max_24h = max((float(t.get('amount', 0)) for t in recent_24h), default=0)

    amounts_7d = [float(t.get('amount', 0)) for t in recent_7d]
    avg_7d = np.mean(amounts_7d) if amounts_7d else amount
    amount_vs_avg = amount / (avg_7d + 1)

    unique_cp_24h = len(set(t.get('counterparty_id', '') for t in recent_24h if t.get('counterparty_id')))

    daily_avg_7d = len(recent_7d) / 7.0
    velocity_change = txn_count_24h / (daily_avg_7d + 1)

    # Cross-border flag
    cross_border = 1.0 if 'cross-border' in payment_type else 0.0

    return [
        amount,            # f1
        np.log1p(amount),  # f2
        hour,              # f3
        dow,               # f4
        is_weekend,        # f5
        is_odd,            # f6
        txn_type_encoded,  # f7
        sender_risk,       # f8
        receiver_risk,     # f9
        currency_mismatch, # f10
        txn_count_24h,     # f11
        total_24h,         # f12
        avg_7d,            # f13
        amount_vs_avg,     # f14
        unique_cp_24h,     # f15
        max_24h,           # f16
        velocity_change,   # f17
        cross_border       # f18
    ]


# ── Scoring ───────────────────────────────────────────────────────────────

def run_isolation_forest(transaction, history=None):
    """Score transaction using trained Isolation Forest"""
    features = build_transaction_features(transaction, history)
    clf = _get_classifier()

    sample = np.array(features).reshape(1, -1)
    raw_score = float(clf.decision_function(sample)[0])

    # Map to 0-1 scale (1 = anomalous)
    anomaly_score = float(max(0.0, min(1.0, 0.5 - raw_score)))
    clean_score = round(1.0 - anomaly_score, 4)

    return {
        'anomaly_score': round(anomaly_score, 4),
        'clean_score': clean_score,
        'is_anomaly': anomaly_score > 0.60,
        'feature_vector': features,
        'raw_score': round(raw_score, 4),
    }


def _build_synthetic_population(n_samples=500):
    """Fallback: synthetic data if no trained model"""
    np.random.seed(42)

    amounts = np.random.lognormal(mean=8.5, sigma=1.2, size=n_samples)
    log_amounts = np.log1p(amounts)
    hours = np.random.choice(range(9, 22), size=n_samples)
    dows = np.random.choice(range(7), size=n_samples, p=[0.18]*5 + [0.05, 0.05])
    is_weekends = (dows >= 5).astype(int)
    is_odds = np.zeros(n_samples, dtype=int)
    txn_types = np.random.uniform(0.2, 0.6, n_samples)
    sender_risks = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    receiver_risks = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    currency_mismatches = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    counts_24h = np.random.poisson(lam=3, size=n_samples).clip(0, 15)
    totals_24h = amounts * counts_24h * 0.7
    avgs_7d = amounts * np.random.uniform(0.7, 1.3, n_samples)
    amt_vs_avg = amounts / (avgs_7d + 1)
    unique_cp = np.random.poisson(lam=2, size=n_samples).clip(1, 8)
    max_24h = amounts * np.random.uniform(0.8, 2.0, n_samples)
    velocities = np.random.uniform(0.5, 2.5, n_samples)
    cross_borders = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    return np.column_stack([
        amounts, log_amounts, hours, dows, is_weekends, is_odds,
        txn_types, sender_risks, receiver_risks, currency_mismatches,
        counts_24h, totals_24h, avgs_7d, amt_vs_avg, unique_cp,
        max_24h, velocities, cross_borders
    ])


def _parse_time(ts):
    if ts is None:
        return datetime.now()
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except:
        return datetime.now()
