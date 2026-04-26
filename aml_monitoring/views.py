"""AML monitoring API views.

Streamlit calls POST /aml/monitor/ with a JSON payload and expects a JSON
response containing aml_score, verdict, rule_flags, etc.
"""

from __future__ import annotations

from typing import Any, Dict

from rest_framework.decorators import api_view
from rest_framework.response import Response

from .services.aml_decision_service import run_aml_monitoring


_COUNTRY_TO_LOCATION = {
    "IN": "India",
    "US": "United States",
    "GB": "United Kingdom",
    "AE": "UAE",
    "SG": "Singapore",
    "IR": "Iran",
    "KP": "North Korea",
    "SY": "Syria",
    "PK": "Pakistan",
    "AF": "Afghanistan",
}


def _recommended_action(verdict: str) -> str:
    verdict_u = (verdict or "").upper()
    if verdict_u == "CLEAN":
        return "APPROVE"
    if verdict_u == "SUSPICIOUS":
        return "REJECT"
    return "MANUAL_REVIEW"


@api_view(["POST"])
def aml_monitor_view(request):
    payload: Dict[str, Any] = request.data if isinstance(request.data, dict) else {}

    transaction: Dict[str, Any] = payload.get("transaction") or {}

    # Streamlit sends ISO timestamps with a trailing 'Z'. Our services use
    # datetime.fromisoformat(), which doesn't accept 'Z' in many Python versions.
    ts = transaction.get("timestamp")
    if isinstance(ts, str) and ts.endswith("Z"):
        transaction["timestamp"] = ts[:-1] + "+00:00"

    # Map UI country codes to the dataset's expected location names.
    cp_country = (transaction.get("counterparty_country") or "").upper()
    if cp_country and not transaction.get("receiver_bank_location"):
        transaction["receiver_bank_location"] = _COUNTRY_TO_LOCATION.get(cp_country, cp_country)

    result = run_aml_monitoring(
        transaction,
        history=payload.get("history") or [],
        account_age_days=payload.get("account_age_days"),
        days_since_last_txn=payload.get("days_since_last_txn"),
    )

    verdict = result.get("verdict")
    if isinstance(verdict, (list, tuple)):
        verdict = verdict[0] if verdict else "MONITOR"

    # Streamlit expects rule_flags like: {code, severity, detail}
    rule_details = result.get("rule_details") or []
    rule_flags = [
        {
            "code": f.get("rule", "--"),
            "severity": f.get("severity", "LOW"),
            "detail": f.get("detail", "--"),
        }
        for f in rule_details
        if isinstance(f, dict)
    ]

    response = {
        "aml_score": result.get("aml_score"),
        "verdict": (verdict or "MONITOR"),
        "rule_flags": rule_flags,
        "isolation_forest_score": result.get("isolation_forest_score"),
        "lstm_anomaly_score": result.get("lstm_anomaly_score"),
        "recommended_action": _recommended_action(str(verdict or "")),
        "score_breakdown": result.get("score_breakdown"),
    }
    return Response(response)
