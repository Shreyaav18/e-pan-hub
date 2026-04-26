"""
Rule-Based Compliance Engine
------------------------------
Implements paper Section V.E.1 — Rule-Based Compliance Engine.

Encodes regulatory rules from FATF, FinCEN, and RBI (India) as
conditional logic. Each rule returns a flag if triggered.

Rules implemented:
  R1  — Large cash transaction above threshold
  R2  — Structuring (multiple just-below-threshold transactions)
  R3  — High-frequency transactions in short window
  R4  — Cross-border transfer to high-risk jurisdiction
  R5  — Unusual transaction time (late night / odd hours)
  R6  — Round-number transactions (common in laundering)
  R7  — Rapid fund movement (in and out quickly)
  R8  — New account high-value activity
  R9  — Multiple counterparties in short window
  R10 — Dormant account sudden activity
"""

from datetime import datetime, timedelta


# ── Thresholds (RBI + FATF aligned) ─────────────────────────────────────────
LARGE_TRANSACTION_THRESHOLD = 50_000      # INR 50,000 CTR threshold
STRUCTURING_THRESHOLD        = 49_000      # Just below CTR
STRUCTURING_WINDOW_HOURS     = 24
HIGH_FREQ_COUNT              = 10          # transactions per hour
ROUND_NUMBER_TOLERANCE       = 0.01        # 1% tolerance for round numbers
RAPID_MOVEMENT_HOURS         = 2           # in-out within 2 hours

# High-risk jurisdictions (LOCATIONS - not ISO codes, from SAML-D dataset)
HIGH_RISK_JURISDICTIONS = {
    'Iran', 'North Korea', 'Syria', 'Myanmar', 'Yemen',
    'Iraq', 'Libya', 'Somalia', 'Pakistan', 'Afghanistan',
    'South Sudan', 'Central African Republic', 'Democratic Republic of Congo',
    'Mali', 'Nicaragua', 'Mexico', 'Turkey', 'Morocco', 'UAE',
    'Tajikistan', 'Uzbekistan'
}

# Odd hours: 11pm – 5am
ODD_HOUR_START = 23
ODD_HOUR_END   = 5


# ── Individual rule checkers ────────────────────────────────────────────────

def rule_large_transaction(txn):
    """R1: Single transaction exceeds CTR threshold."""
    if txn.get('amount', 0) >= LARGE_TRANSACTION_THRESHOLD:
        return {
            'rule':     'R1_LARGE_TRANSACTION',
            'severity': 'HIGH',
            'detail':   f"Transaction amount {txn['amount']:,.0f} exceeds threshold {LARGE_TRANSACTION_THRESHOLD:,}",
        }
    return None


def rule_structuring(txn, history):
    """
    R2: Multiple transactions just below threshold within 24h.
    Classic structuring (smurfing) pattern.
    """
    if txn.get('amount', 0) > STRUCTURING_THRESHOLD:
        return None  # too large to be structuring

    cutoff = _parse_time(txn.get('timestamp')) - timedelta(hours=STRUCTURING_WINDOW_HOURS)
    recent = [
        t for t in history
        if _parse_time(t.get('timestamp')) >= cutoff
        and t.get('amount', 0) > STRUCTURING_THRESHOLD * 0.80
        and t.get('amount', 0) <= STRUCTURING_THRESHOLD
    ]

    if len(recent) >= 3:
        total = sum(t['amount'] for t in recent) + txn['amount']
        return {
            'rule':     'R2_STRUCTURING',
            'severity': 'HIGH',
            'detail':   f"{len(recent)+1} near-threshold transactions totalling {total:,.0f} in 24h window",
        }
    return None


def rule_high_frequency(txn, history):
    """R3: Too many transactions in a 1-hour window."""
    cutoff = _parse_time(txn.get('timestamp')) - timedelta(hours=1)
    recent_count = sum(
        1 for t in history
        if _parse_time(t.get('timestamp')) >= cutoff
    )
    if recent_count >= HIGH_FREQ_COUNT:
        return {
            'rule':     'R3_HIGH_FREQUENCY',
            'severity': 'MEDIUM',
            'detail':   f"{recent_count} transactions in the last hour",
        }
    return None


def rule_high_risk_jurisdiction(txn):
    """R4: Transfer to/from a high-risk jurisdiction (LOCATION NAMES, not ISO)."""
    sender_loc = str(txn.get('sender_bank_location', '')).strip()
    receiver_loc = str(txn.get('receiver_bank_location', '')).strip()

    if sender_loc in HIGH_RISK_JURISDICTIONS:
        return {
            'rule':     'R4_HIGH_RISK_JURISDICTION',
            'severity': 'HIGH',
            'detail':   f"Sender location is high-risk: {sender_loc}",
        }

    if receiver_loc in HIGH_RISK_JURISDICTIONS:
        return {
            'rule':     'R4_HIGH_RISK_JURISDICTION',
            'severity': 'HIGH',
            'detail':   f"Receiver location is high-risk: {receiver_loc}",
        }

    return None


def rule_unusual_time(txn):
    """R5: Transaction at unusual hours (11pm–5am)."""
    ts = _parse_time(txn.get('timestamp'))
    hour = ts.hour
    if hour >= ODD_HOUR_START or hour < ODD_HOUR_END:
        return {
            'rule':     'R5_UNUSUAL_TIME',
            'severity': 'LOW',
            'detail':   f"Transaction at {ts.strftime('%H:%M')} (odd hours)",
        }
    return None


def rule_round_number(txn):
    """
    R6: Suspiciously round transaction amounts.
    e.g. exactly 10000, 50000, 100000.
    """
    amount = txn.get('amount', 0)
    if amount < 1000:
        return None
    # Check if amount is within 1% of a round number
    for base in [1000, 5000, 10000, 25000, 50000, 100000, 500000]:
        if abs(amount - base) / base < ROUND_NUMBER_TOLERANCE:
            return {
                'rule':     'R6_ROUND_NUMBER',
                'severity': 'LOW',
                'detail':   f"Suspiciously round amount: {amount:,.0f}",
            }
    return None


def rule_rapid_movement(txn, history):
    """
    R7: Funds received and sent out within a short window.
    Classic layering technique.
    """
    txn_time = _parse_time(txn.get('timestamp'))
    txn_type = txn.get('type', '').lower()

    if txn_type not in ('debit', 'withdrawal', 'transfer_out'):
        return None

    cutoff = txn_time - timedelta(hours=RAPID_MOVEMENT_HOURS)
    recent_credits = [
        t for t in history
        if _parse_time(t.get('timestamp')) >= cutoff
        and t.get('type', '').lower() in ('credit', 'deposit', 'transfer_in')
        and t.get('amount', 0) > txn.get('amount', 0) * 0.80
    ]

    if recent_credits:
        return {
            'rule':     'R7_RAPID_MOVEMENT',
            'severity': 'HIGH',
            'detail':   f"Funds credited then debited within {RAPID_MOVEMENT_HOURS}h",
        }
    return None


def rule_new_account_high_value(txn, account_age_days):
    """R8: High-value activity on a newly created account."""
    if account_age_days is not None and account_age_days <= 30:
        if txn.get('amount', 0) >= 25_000:
            return {
                'rule':     'R8_NEW_ACCOUNT_HIGH_VALUE',
                'severity': 'MEDIUM',
                'detail':   f"Account {account_age_days} days old with transaction {txn['amount']:,.0f}",
            }
    return None


def rule_multiple_counterparties(txn, history):
    """R9: Transactions with many different counterparties in 24h."""
    cutoff = _parse_time(txn.get('timestamp')) - timedelta(hours=24)
    counterparties = set(
        t.get('counterparty_id', '')
        for t in history
        if _parse_time(t.get('timestamp')) >= cutoff
        and t.get('counterparty_id')
    )
    counterparties.add(txn.get('counterparty_id', ''))

    if len(counterparties) >= 8:
        return {
            'rule':     'R9_MULTIPLE_COUNTERPARTIES',
            'severity': 'MEDIUM',
            'detail':   f"{len(counterparties)} unique counterparties in 24h window",
        }
    return None


def rule_dormant_account(txn, days_since_last_txn):
    """R10: Sudden high-value activity after long dormancy."""
    if days_since_last_txn is not None and days_since_last_txn >= 180:
        if txn.get('amount', 0) >= 10_000:
            return {
                'rule':     'R10_DORMANT_ACCOUNT',
                'severity': 'MEDIUM',
                'detail':   f"Account dormant for {days_since_last_txn} days, now showing activity",
            }
    return None


def rule_currency_mismatch(txn):
    """R11: Payment currency != Received currency (only flag if huge amount)."""
    payment_curr = str(txn.get('payment_currency', '')).upper()
    received_curr = str(txn.get('received_currency', '')).upper()
    amount = txn.get('amount', 0)

    # ONLY flag if huge amount (> 100,000)
    if payment_curr != received_curr and amount > 100_000:
        return {
            'rule':     'R11_CURRENCY_MISMATCH',
            'severity': 'MEDIUM',
            'detail':   f"Currency mismatch {payment_curr}→{received_curr} with large amount {amount:,.0f}",
        }
    return None


def rule_cross_border_payment(txn):
    """R12: Cross-border payment explicitly flagged in dataset."""
    payment_type = str(txn.get('payment_type', '')).lower()

    if 'cross-border' in payment_type:
        return {
            'rule':     'R12_CROSS_BORDER_PAYMENT',
            'severity': 'LOW',
            'detail':   'Cross-border payment (requires monitoring)',
        }
    return None


# ── Master rule runner ────────────────────────────────────────────────────────

ALL_RULES = [
    rule_large_transaction,
    rule_high_risk_jurisdiction,
    rule_unusual_time,
    rule_round_number,
    rule_currency_mismatch,        # NEW - R11
    rule_cross_border_payment,     # NEW - R12
]

HISTORY_RULES = [
    rule_structuring,
    rule_high_frequency,
    rule_rapid_movement,
    rule_multiple_counterparties,
]


def run_rule_engine(transaction, history=None, account_age_days=None,
                    days_since_last_txn=None):
    """
    Run all FATF/RBI compliance rules against a transaction.

    Parameters
    ----------
    transaction          : dict — single transaction to evaluate
    history              : list of prior transaction dicts
    account_age_days     : int or None
    days_since_last_txn  : int or None

    Returns
    -------
    dict:
        flags            : list of triggered rule dicts
        severity_counts  : {'HIGH': n, 'MEDIUM': n, 'LOW': n}
        rule_score       : float 0–1 (1 = clean, 0 = many high-severity flags)
        has_hard_violation: bool (any HIGH severity)
    """
    history = history or []
    flags   = []

    # Single-transaction rules
    for rule_fn in ALL_RULES:
        result = rule_fn(transaction)
        if result:
            flags.append(result)

    # History-dependent rules
    for rule_fn in HISTORY_RULES:
        result = rule_fn(transaction, history)
        if result:
            flags.append(result)

    # Context rules
    if account_age_days is not None:
        r = rule_new_account_high_value(transaction, account_age_days)
        if r:
            flags.append(r)

    if days_since_last_txn is not None:
        r = rule_dormant_account(transaction, days_since_last_txn)
        if r:
            flags.append(r)

    # Severity counts
    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for f in flags:
        severity_counts[f['severity']] += 1

    # Rule score: penalise by severity
    # HIGH costs 0.30, MEDIUM 0.15, LOW 0.05
    penalty = (
        severity_counts['HIGH']   * 0.30 +
        severity_counts['MEDIUM'] * 0.15 +
        severity_counts['LOW']    * 0.05
    )
    rule_score = max(0.0, round(1.0 - penalty, 4))

    return {
        'flags':              flags,
        'severity_counts':    severity_counts,
        'rule_score':         rule_score,
        'has_hard_violation': severity_counts['HIGH'] > 0,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_time(ts):
    """Parse ISO string or return now() if None."""
    if ts is None:
        return datetime.now()
    if isinstance(ts, datetime):
        return ts
    try:
        s = str(ts)
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now()