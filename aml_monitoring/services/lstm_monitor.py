"""
LSTM Sequential Pattern Monitor
----------------------------------
Implements paper Section V.E.2 — LSTM for Sequential Pattern Detection.

Paper equation (9):
    AnomalyScore_LSTM = ||x_actual - x_predicted||_2

The LSTM is trained to predict the next transaction's features given
a sequence of prior transactions. A large deviation from prediction
indicates an unexpected pattern — potential money laundering.

Implementation note:
    Full LSTM training requires PyTorch/TensorFlow + significant data.
    We implement:
      1. A statistical sequence model (ARIMA-style) as a lightweight
         stand-in for the LSTM prediction step — same equation (9) logic.
      2. A feature-based sequential anomaly scorer that implements the
         same paper logic without the heavy DL dependency.
    This matches the paper's intent while keeping the project runnable.
"""

import numpy as np
from datetime import datetime, timedelta


# ── Sequence feature extraction ──────────────────────────────────────────────

def extract_sequence_features(history, window=10):
    """
    Build a time-ordered feature matrix from transaction history.
    Each row = one transaction's features (7 dimensions):
        [log_amount, hour, txn_type, is_weekend, cp_risk, currency_mismatch, cross_border]

    Returns numpy array of shape (min(len(history), window), 7).
    """
    if not history:
        return np.array([]).reshape(0, 7)

    # Sort by timestamp
    sorted_txns = sorted(history, key=lambda t: _parse_time(t.get('timestamp')))
    recent = sorted_txns[-window:]

    type_map = {'credit': 0, 'deposit': 0, 'transfer_in': 0,
                'debit': 1, 'withdrawal': 1, 'transfer_out': 1,
                'transfer': 2}
    high_risk_locations = {
        'Iran', 'North Korea', 'Syria', 'Myanmar', 'Yemen',
        'Iraq', 'Libya', 'Somalia', 'Pakistan', 'Afghanistan',
        'South Sudan', 'Central African Republic', 'Democratic Republic of Congo',
        'Mali', 'Nicaragua', 'Mexico', 'Turkey', 'Morocco', 'UAE'
    }

    rows = []
    for t in recent:
        ts = _parse_time(t.get('timestamp'))
        amount = float(t.get('amount', 0))
        payment_type = str(t.get('payment_type', 'other')).lower()
        payment_curr = str(t.get('payment_currency', '')).upper()
        received_curr = str(t.get('received_currency', '')).upper()
        receiver_loc = str(t.get('receiver_bank_location', '')).strip()

        rows.append([
            np.log1p(amount),
            ts.hour / 23.0,
            type_map.get(t.get('type', '').lower(), 2) / 2.0,
            float(ts.weekday() >= 5),
            float(receiver_loc in high_risk_locations),
            float(payment_curr != received_curr),  # currency_mismatch
            float('cross-border' in payment_type),  # cross_border
        ])

    return np.array(rows)


# ── Statistical sequence predictor ───────────────────────────────────────────

def predict_next_features(sequence_matrix):
    """
    Predict expected features of the next transaction using
    exponential weighted moving average — a lightweight proxy
    for the LSTM prediction step.

    In production: replace with a trained LSTM that does:
        hidden_state, cell_state = lstm(sequence)
        predicted = linear_layer(hidden_state)
    """
    if len(sequence_matrix) == 0:
        return None

    # Exponential weights — recent transactions matter more
    n       = len(sequence_matrix)
    weights = np.exp(np.linspace(0, 1, n))
    weights /= weights.sum()

    predicted = np.average(sequence_matrix, axis=0, weights=weights)
    return predicted


# ── Paper equation (9): AnomalyScore = ||x_actual - x_predicted||_2 ─────────

def compute_lstm_anomaly_score(transaction, history=None):
    """
    Implements paper equation (9):
        AnomalyScore_LSTM = ||x_actual - x_predicted||_2

    Parameters
    ----------
    transaction : dict — the transaction to evaluate
    history     : list of prior transactions for sequence context

    Returns
    -------
    dict:
        lstm_anomaly_score : float 0–1 (0 = expected, 1 = highly anomalous)
        l2_distance        : raw Euclidean distance
        predicted_features : what was expected
        actual_features    : what was observed
        sequential_verdict : str
    """
    history = history or []

    # Build sequence from history
    sequence = extract_sequence_features(history, window=10)

    # Build actual feature vector for current transaction (7 dimensions)
    high_risk_locations = {
        'Iran', 'North Korea', 'Syria', 'Myanmar', 'Yemen',
        'Iraq', 'Libya', 'Somalia', 'Pakistan', 'Afghanistan',
        'South Sudan', 'Central African Republic', 'Democratic Republic of Congo',
        'Mali', 'Nicaragua', 'Mexico', 'Turkey', 'Morocco', 'UAE'
    }
    type_map = {'credit': 0, 'deposit': 0, 'transfer_in': 0,
                'debit': 1, 'withdrawal': 1, 'transfer_out': 1,
                'transfer': 2}

    ts = _parse_time(transaction.get('timestamp'))
    amount = float(transaction.get('amount', 0))
    payment_type = str(transaction.get('payment_type', 'other')).lower()
    payment_curr = str(transaction.get('payment_currency', '')).upper()
    received_curr = str(transaction.get('received_currency', '')).upper()
    receiver_loc = str(transaction.get('receiver_bank_location', '')).strip()

    actual = np.array([
        np.log1p(amount),
        ts.hour / 23.0,
        type_map.get(transaction.get('type', '').lower(), 2) / 2.0,
        float(ts.weekday() >= 5),
        float(receiver_loc in high_risk_locations),
        float(payment_curr != received_curr),  # currency_mismatch
        float('cross-border' in payment_type),  # cross_border
    ])

    # If no history: no sequential anomaly possible
    if len(sequence) == 0:
        return {
            'lstm_anomaly_score': 0.3,   # neutral
            'l2_distance':        0.0,
            'predicted_features': actual.tolist(),
            'actual_features':    actual.tolist(),
            'sequential_verdict': 'INSUFFICIENT_HISTORY',
        }

    # Predict next features using sequence
    predicted = predict_next_features(sequence)

    # Paper equation (9): L2 norm of deviation
    l2_distance = float(np.linalg.norm(actual - predicted))

    # Normalise: typical L2 in feature space is 0–3
    # Scale: 0 → score 0.0 (expected), 3+ → score 1.0 (highly anomalous)
    max_expected_distance = 2.5
    lstm_anomaly_score = min(1.0, l2_distance / max_expected_distance)
    lstm_anomaly_score = round(float(lstm_anomaly_score), 4)

    # Sequential verdict
    if lstm_anomaly_score < 0.30:
        verdict = 'EXPECTED_PATTERN'
    elif lstm_anomaly_score < 0.60:
        verdict = 'SLIGHT_DEVIATION'
    else:
        verdict = 'PATTERN_BREAK'

    return {
        'lstm_anomaly_score': lstm_anomaly_score,
        'l2_distance':        round(l2_distance, 4),
        'predicted_features': predicted.tolist(),
        'actual_features':    actual.tolist(),
        'sequential_verdict': verdict,
    }


# ── Pattern-level checks ──────────────────────────────────────────────────────

def detect_layering_pattern(history):
    """
    Detect classic layering: rapid in → multiple outs to different parties.
    Returns a risk score 0–1.
    """
    if len(history) < 4:
        return 0.0

    sorted_txns = sorted(history, key=lambda t: _parse_time(t.get('timestamp')))

    # Look for a large credit followed by multiple debits in 48h
    for i, txn in enumerate(sorted_txns):
        if txn.get('type', '').lower() not in ('credit', 'deposit', 'transfer_in'):
            continue

        credit_amount = float(txn.get('amount', 0))
        if credit_amount < 10_000:
            continue

        credit_time = _parse_time(txn.get('timestamp'))
        cutoff = credit_time + timedelta(hours=48)

        subsequent_debits = [
            t for t in sorted_txns[i + 1:]
            if _parse_time(t.get('timestamp')) <= cutoff
            and t.get('type', '').lower() in ('debit', 'withdrawal', 'transfer_out')
        ]

        if len(subsequent_debits) >= 3:
            unique_cps = len(set(t.get('counterparty_id', '') for t in subsequent_debits))
            if unique_cps >= 2:
                # Score: more splits = more suspicious
                return min(1.0, 0.3 + (unique_cps - 1) * 0.15)

    return 0.0


def detect_smurfing_pattern(history):
    """
    Detect smurfing: many small transactions that aggregate above threshold.
    Returns risk score 0–1.
    """
    if len(history) < 5:
        return 0.0

    threshold    = 50_000
    window_hours = 48

    sorted_txns  = sorted(history, key=lambda t: _parse_time(t.get('timestamp')))

    for i in range(len(sorted_txns)):
        start_time = _parse_time(sorted_txns[i].get('timestamp'))
        end_time   = start_time + timedelta(hours=window_hours)

        window_txns = [
            t for t in sorted_txns[i:]
            if _parse_time(t.get('timestamp')) <= end_time
            and t.get('type', '').lower() in ('credit', 'deposit', 'transfer_in')
        ]

        if len(window_txns) >= 5:
            total = sum(float(t.get('amount', 0)) for t in window_txns)
            if total >= threshold:
                # Check that individual amounts were all below threshold
                all_below = all(float(t.get('amount', 0)) < threshold for t in window_txns)
                if all_below:
                    return min(1.0, 0.5 + (len(window_txns) - 5) * 0.05)

    return 0.0


# ── Master LSTM monitor ───────────────────────────────────────────────────────

def run_lstm_monitor(transaction, history=None):
    """
    Full LSTM sequential monitoring pipeline.

    Returns
    -------
    dict:
        lstm_anomaly_score  : float 0–1 (combined sequential score)
        sequence_score      : float 0–1 (1 = clean sequential pattern)
        layering_score      : float 0–1
        smurfing_score      : float 0–1
        sequential_verdict  : str
        details             : dict of sub-scores
    """
    history = history or []

    # Core LSTM prediction deviation (paper equation 9)
    lstm_result = compute_lstm_anomaly_score(transaction, history)

    # Pattern-level checks on full history
    layering_score = detect_layering_pattern(history)
    smurfing_score = detect_smurfing_pattern(history)

    # Combined sequential anomaly score
    combined = (
        lstm_result['lstm_anomaly_score'] * 0.50 +
        layering_score                    * 0.30 +
        smurfing_score                    * 0.20
    )
    combined = round(float(min(1.0, combined)), 4)

    # sequence_score: consistent direction (1 = safe)
    sequence_score = round(1.0 - combined, 4)

    # Final verdict
    if combined < 0.25:
        verdict = 'CLEAN'
    elif combined < 0.55:
        verdict = 'MONITOR'
    else:
        verdict = 'SUSPICIOUS_PATTERN'

    return {
        'lstm_anomaly_score': combined,
        'sequence_score':     sequence_score,
        'layering_score':     layering_score,
        'smurfing_score':     smurfing_score,
        'sequential_verdict': verdict,
        'details': {
            'l2_deviation':    lstm_result['l2_distance'],
            'pattern_verdict': lstm_result['sequential_verdict'],
        },
    }


def _parse_time(ts):
    if ts is None:
        return datetime.now()
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return datetime.now()