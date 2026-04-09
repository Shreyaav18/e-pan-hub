"""
AML Decision Service
----------------------
Orchestrates all three AML sub-services and produces
the final SAML score saved to VerificationCase.

Hybrid approach (paper Section V.E):
    Rule engine    → hard violations + rule_score
    Isolation Forest → statistical anomaly score
    LSTM monitor   → sequential pattern score

Final SAML = weighted combination of all three,
with hard overrides for HIGH-severity rule violations.
"""

from .rule_engine       import run_rule_engine
from .isolation_forest  import run_isolation_forest
from .lstm_monitor      import run_lstm_monitor


# Weights for final AML score
AML_WEIGHTS = {
    'rule':    0.40,   # Rule engine is the regulatory baseline
    'iforest': 0.35,   # Statistical anomaly
    'lstm':    0.25,   # Sequential patterns
}


def run_aml_monitoring(transaction, history=None,
                       account_age_days=None,
                       days_since_last_txn=None,
                       payment_currency=None,
                       received_currency=None,
                       receiver_bank_location=None,
                       payment_type=None):
    """
    Full AML monitoring pipeline for a single transaction.

    Parameters
    ----------
    transaction          : dict with keys:
                              amount, type, timestamp,
                              counterparty_id, counterparty_country
    history              : list of prior transaction dicts
    account_age_days     : int — how old is this account
    days_since_last_txn  : int — days since last activity

    Returns
    -------
    dict with all fields for VerificationCase + display context
    """
    history = history or []
    # Merge SAML-D fields into transaction dict if passed separately
    if payment_currency:
        transaction['payment_currency'] = payment_currency
    if received_currency:
        transaction['received_currency'] = received_currency
    if receiver_bank_location:
        transaction['receiver_bank_location'] = receiver_bank_location
    if payment_type:
        transaction['payment_type'] = payment_type

    # ── Step 1: Rule engine ───────────────────────────────────────────────
    rule_result = run_rule_engine(
        transaction,
        history=history,
        account_age_days=account_age_days,
        days_since_last_txn=days_since_last_txn,
    )

    # ── Step 2: Isolation Forest ──────────────────────────────────────────
    iforest_result = run_isolation_forest(transaction, history)

    # ── Step 3: LSTM sequential monitor ──────────────────────────────────
    lstm_result = run_lstm_monitor(transaction, history)

    # ── Step 4: Weighted SAML ─────────────────────────────────────────────
    rule_score   = rule_result['rule_score']
    iforest_clean = iforest_result['clean_score']
    lstm_clean    = lstm_result['sequence_score']

    saml = (
        AML_WEIGHTS['rule']    * rule_score    +
        AML_WEIGHTS['iforest'] * iforest_clean +
        AML_WEIGHTS['lstm']    * lstm_clean
    )
    saml = round(float(min(1.0, max(0.0, saml))), 4)

    # ── Step 5: Hard override for HIGH severity rules ─────────────────────
    penalty_flags = []
    if rule_result['has_hard_violation']:
        saml = min(saml, 0.30)  # cap score if hard rule triggered
        penalty_flags.append('AML_HARD_RULE_VIOLATION')

    # ── Step 6: Map laundering type (from SAML-D dataset) ─────────────────
    from aml_monitoring.utils import map_laundering_type, get_laundering_description
    laundering_type = transaction.get('laundering_type', '')
    applicable_rules = map_laundering_type(laundering_type)
    laundering_desc = get_laundering_description(laundering_type) if laundering_type else ''

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict = _aml_verdict(saml)

    return {
        # DB fields
        'rule_flags':             [f['rule'] for f in rule_result['flags']],
        'isolation_forest_score': iforest_result['anomaly_score'],
        'lstm_anomaly_score':     lstm_result['lstm_anomaly_score'],
        'aml_score':              saml,
        'penalty_flags':          penalty_flags,
        'laundering_type':        laundering_type,

        # Display context
        'rule_details':           rule_result['flags'],
        'severity_counts':        rule_result['severity_counts'],
        'iforest_is_anomaly':     iforest_result['is_anomaly'],
        'lstm_verdict':           lstm_result['sequential_verdict'],
        'layering_score':         lstm_result['layering_score'],
        'smurfing_score':         lstm_result['smurfing_score'],
        'applicable_rules':       applicable_rules,
        'laundering_description': laundering_desc,
        'verdict':                verdict,
        'has_hard_violation':     rule_result['has_hard_violation'],

        # Score breakdown for UI
        'score_breakdown': {
            'Rule Engine':      rule_score,
            'Isolation Forest': iforest_clean,
            'LSTM Sequential':  lstm_clean,
        },
    }


def _aml_verdict(saml):
    if saml >= 0.70:
        return ('CLEAN',     'success')
    elif saml >= 0.40:
        return ('MONITOR',   'warning')
    else:
        return ('SUSPICIOUS', 'danger')