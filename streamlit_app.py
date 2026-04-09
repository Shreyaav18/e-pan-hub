import streamlit as st
import requests
import json
import time
import random
from datetime import datetime, timedelta

st.set_page_config(
    page_title="KYC Verification System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #06070A !important;
    color: #C8CFD8 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #0B0D12 !important;
    border-right: 1px solid #1A1E28 !important;
}

[data-testid="stSidebar"] * { color: #C8CFD8 !important; }

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #0B0D12 !important;
    border-bottom: 1px solid #1A1E28 !important;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    color: #5A6377 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    color: #3ECFCF !important;
    border-bottom: 2px solid #3ECFCF !important;
    background-color: transparent !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid #3ECFCF !important;
    color: #3ECFCF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    border-radius: 2px !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background: #3ECFCF !important;
    color: #06070A !important;
    box-shadow: 0 0 24px rgba(62, 207, 207, 0.25) !important;
}

.stFileUploader {
    border: 1px dashed #1E2430 !important;
    border-radius: 4px !important;
    background: #0B0D12 !important;
    transition: border-color 0.2s ease;
}

.stFileUploader:hover { border-color: #3ECFCF !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stDateInput > div > div > input {
    background-color: #0F1117 !important;
    border: 1px solid #1A1E28 !important;
    border-radius: 2px !important;
    color: #C8CFD8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #3ECFCF !important;
    box-shadow: 0 0 0 1px #3ECFCF !important;
}

.stSelectbox > div > div { background-color: #0F1117 !important; }

[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #3ECFCF !important;
    font-size: 28px !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5A6377 !important;
}

.stProgress > div > div > div {
    background-color: #3ECFCF !important;
    border-radius: 0 !important;
}

.stProgress > div > div {
    background-color: #1A1E28 !important;
    border-radius: 0 !important;
    height: 3px !important;
}

.verdict-card {
    background: #0B0D12;
    border: 1px solid #1A1E28;
    border-radius: 4px;
    padding: 24px;
    margin: 8px 0;
    animation: fadeSlideIn 0.4s ease forwards;
    opacity: 0;
}

.verdict-card.approved {
    border-left: 3px solid #00D68F;
    animation-delay: 0.1s;
}

.verdict-card.rejected {
    border-left: 3px solid #FF4757;
    animation-delay: 0.1s;
}

.verdict-card.review {
    border-left: 3px solid #FFA502;
    animation-delay: 0.1s;
}

.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1A1E28;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    animation: fadeSlideIn 0.4s ease forwards;
    opacity: 0;
}

.score-row:last-child { border-bottom: none; }

.score-label { color: #5A6377; letter-spacing: 0.05em; }

.score-value { 
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 2px;
}

.score-high { color: #00D68F; background: rgba(0,214,143,0.08); }
.score-mid  { color: #FFA502; background: rgba(255,165,2,0.08); }
.score-low  { color: #FF4757; background: rgba(255,71,87,0.08); }

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3ECFCF;
    padding-bottom: 8px;
    border-bottom: 1px solid #1A1E28;
    margin-bottom: 16px;
}

.rule-flag {
    background: #0F1117;
    border: 1px solid #1A1E28;
    border-radius: 2px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    animation: fadeSlideIn 0.4s ease forwards;
    opacity: 0;
}

.rule-flag.high { border-left: 2px solid #FF4757; }
.rule-flag.medium { border-left: 2px solid #FFA502; }
.rule-flag.low { border-left: 2px solid #5A6377; }

.rule-code { color: #FF4757; font-weight: 500; margin-bottom: 4px; }
.rule-code.medium { color: #FFA502; }
.rule-code.low { color: #5A6377; }
.rule-detail { color: #5A6377; font-size: 10px; }

.dashboard-stat {
    background: #0B0D12;
    border: 1px solid #1A1E28;
    padding: 20px;
    border-radius: 2px;
    text-align: center;
    transition: border-color 0.2s ease;
}

.dashboard-stat:hover { border-color: #3ECFCF; }

.stat-number {
    font-family: 'DM Mono', monospace;
    font-size: 36px;
    font-weight: 500;
    color: #3ECFCF;
    display: block;
    line-height: 1;
}

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5A6377;
    margin-top: 8px;
    display: block;
}

.case-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 0;
    border-bottom: 1px solid #1A1E28;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    animation: fadeSlideIn 0.3s ease forwards;
    opacity: 0;
}

.status-pill {
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.status-approved { background: rgba(0,214,143,0.12); color: #00D68F; }
.status-rejected { background: rgba(255,71,87,0.12); color: #FF4757; }
.status-review   { background: rgba(255,165,2,0.12); color: #FFA502; }
.status-pending  { background: rgba(90,99,119,0.12); color: #5A6377; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #3ECFCF;
    margin-bottom: 2px;
}

.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: #5A6377;
}

.divider {
    border: none;
    border-top: 1px solid #1A1E28;
    margin: 20px 0;
}

.aml-score-ring {
    width: 100%;
    text-align: center;
    padding: 24px 0;
}

.score-circle-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5A6377;
    margin-top: 12px;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(62,207,207,0.15); }
    50%       { box-shadow: 0 0 24px rgba(62,207,207,0.35); }
}

.glow-box {
    animation: pulse-glow 3s ease-in-out infinite;
    border-color: #3ECFCF !important;
}

.sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3ECFCF;
    margin-bottom: 8px;
    display: block;
}

.stSlider > div { color: #5A6377 !important; }
.stCheckbox > label { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }
[data-testid="stForm"] { background: transparent !important; border: none !important; }
div[data-testid="column"] { padding: 0 8px; }

.upload-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5A6377;
    margin-bottom: 6px;
}

.info-bar {
    background: #0B0D12;
    border: 1px solid #1A1E28;
    border-left: 2px solid #3ECFCF;
    padding: 10px 16px;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #5A6377;
    margin: 12px 0;
}

.stSpinner > div { border-top-color: #3ECFCF !important; }

div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000"

def score_color_class(score):
    if score >= 0.75:
        return "score-high"
    elif score >= 0.50:
        return "score-mid"
    return "score-low"

def score_to_pct(score):
    return f"{score * 100:.1f}%"

def get_verdict_class(status):
    mapping = {"approved": "approved", "rejected": "rejected", "review": "review", "manual_review": "review"}
    return mapping.get(status.lower(), "review")

def mock_doc_result():
    return {
        "case_id": f"CASE-{random.randint(10000,99999)}",
        "status": random.choice(["approved", "review", "rejected"]),
        "doc_score": round(random.uniform(0.65, 0.99), 3),
        "doc_verdict": "AUTHENTIC",
        "doc_ocr_valid": round(random.uniform(0.80, 1.0), 3),
        "doc_font_consistency": round(random.uniform(0.70, 0.98), 3),
        "doc_edge_consistency": round(random.uniform(0.72, 0.96), 3),
        "doc_color_histogram": round(random.uniform(0.68, 0.95), 3),
        "doc_structural_align": round(random.uniform(0.75, 0.97), 3),
        "doc_metadata_clean": round(random.uniform(0.80, 1.0), 3),
        "doc_ssim_score": round(random.uniform(0.60, 0.90), 3),
        "extracted_pan_number": "ABCDE1234F",
        "extracted_name": "RAJESH KUMAR SHARMA",
        "extracted_dob": "15/08/1985",
        "biometric_score": round(random.uniform(0.70, 0.96), 3),
        "face_match_score": round(random.uniform(0.75, 0.98), 3),
        "liveness_score": round(random.uniform(0.72, 0.96), 3),
        "behavioral_score": round(random.uniform(0.55, 0.90), 3),
    }

def mock_aml_result(amount):
    flags = []
    if amount >= 50000:
        flags.append({"code": "R1_LARGE_TRANSACTION", "severity": "HIGH", "detail": f"Amount {amount:,.0f} exceeds CTR threshold"})
    if datetime.now().hour >= 23 or datetime.now().hour <= 5:
        flags.append({"code": "R5_UNUSUAL_TIME", "severity": "LOW", "detail": f"Transaction at {datetime.now().strftime('%H:%M')}"})
    if amount % 1000 == 0:
        flags.append({"code": "R6_ROUND_NUMBER", "severity": "LOW", "detail": f"Amount {amount:,.0f} is suspiciously round"})
    aml_score = max(0.1, 1.0 - (0.30 * sum(1 for f in flags if f["severity"] == "HIGH")) - (0.10 * len(flags)))
    verdict = "CLEAN" if aml_score >= 0.70 else ("MONITOR" if aml_score >= 0.40 else "SUSPICIOUS")
    return {
        "aml_score": round(aml_score, 3),
        "verdict": verdict,
        "rule_flags": flags,
        "isolation_forest_score": round(random.uniform(0.30, 0.75), 3),
        "lstm_anomaly_score": round(random.uniform(0.35, 0.80), 3),
        "recommended_action": "MANUAL_REVIEW" if verdict == "MONITOR" else ("APPROVE" if verdict == "CLEAN" else "REJECT"),
    }

def render_sidebar():
    st.sidebar.markdown('<span class="sidebar-label">System</span>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:16px;font-weight:700;color:#EAEEF4;letter-spacing:-0.01em;">KYC / AML Platform</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#5A6377;margin-top:2px;">v1.0 — Compliance Suite</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<hr style="border-color:#1A1E28;margin:16px 0">', unsafe_allow_html=True)

    st.sidebar.markdown('<span class="sidebar-label">Configuration</span>', unsafe_allow_html=True)
    backend = st.sidebar.text_input("API Endpoint", value=BACKEND_URL, key="backend_url")
    use_mock = st.sidebar.checkbox("Use Mock Data", value=True, help="Enable to test without a running backend")

    st.sidebar.markdown('<hr style="border-color:#1A1E28;margin:16px 0">', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sidebar-label">Thresholds</span>', unsafe_allow_html=True)

    doc_thresh  = st.sidebar.slider("Document Score", 0.0, 1.0, 0.75, 0.01)
    bio_thresh  = st.sidebar.slider("Biometric Score", 0.0, 1.0, 0.75, 0.01)
    aml_thresh  = st.sidebar.slider("AML Score", 0.0, 1.0, 0.70, 0.01)

    st.sidebar.markdown('<hr style="border-color:#1A1E28;margin:16px 0">', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sidebar-label">Session</span>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:#5A6377;">'
        f'Started: {datetime.now().strftime("%d %b %Y  %H:%M")}</div>',
        unsafe_allow_html=True
    )

    return backend, use_mock, doc_thresh, bio_thresh, aml_thresh

def render_score_row(label, value, delay=0):
    cls = score_color_class(value)
    style = f"animation-delay:{delay:.1f}s"
    return f"""
    <div class="score-row" style="{style}">
        <span class="score-label">{label}</span>
        <span class="score-value {cls}">{score_to_pct(value)}</span>
    </div>"""

def render_kyc_tab(use_mock):
    st.markdown('<div class="section-header">Document Verification + Biometric Authentication</div>', unsafe_allow_html=True)

    col_form, col_results = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Submission</div>', unsafe_allow_html=True)

        full_name = st.text_input("Full Name", placeholder="As it appears on PAN card")
        
        st.markdown('<div class="upload-label">PAN Card Image</div>', unsafe_allow_html=True)
        pan_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], key="pan_upload", label_visibility="collapsed")
        
        st.markdown('<div class="upload-label">Live Selfie</div>', unsafe_allow_html=True)
        selfie_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="selfie_upload", label_visibility="collapsed")
        
        st.markdown('<div class="upload-label">Reference PAN (optional)</div>', unsafe_allow_html=True)
        ref_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="ref_upload", label_visibility="collapsed")

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        run_btn = st.button("Run Verification", key="run_kyc")

        if not use_mock:
            st.markdown(
                '<div class="info-bar">Ensure the backend is reachable at the configured API endpoint before submitting.</div>',
                unsafe_allow_html=True
            )

    with col_results:
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Processing verification pipeline..."):
                time.sleep(1.8)

            if use_mock:
                result = mock_doc_result()
            else:
                try:
                    files = {}
                    data = {"full_name": full_name}
                    if pan_file:
                        files["pan_card"] = pan_file
                    if selfie_file:
                        files["selfie_image"] = selfie_file
                    if ref_file:
                        files["reference_card"] = ref_file
                    resp = requests.post(f"{BACKEND_URL}/verify/upload/", data=data, files=files, timeout=30)
                    result = resp.json()
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    return

            st.session_state["kyc_result"] = result

        result = st.session_state.get("kyc_result")

        if result:
            status = result.get("status", "review")
            vclass = get_verdict_class(status)

            final_html = f"""
            <div class="verdict-card {vclass}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5A6377;">Case ID</div>
                        <div style="font-family:'DM Mono',monospace;font-size:13px;color:#EAEEF4;margin-top:2px;">{result.get("case_id","--")}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="status-pill status-{status.lower()}">{status.upper()}</div>
                    </div>
                </div>
            </div>"""
            st.markdown(final_html, unsafe_allow_html=True)

            col_d, col_b = st.columns(2)

            with col_d:
                doc_score = result.get("doc_score", 0)
                st.markdown(f'<div class="section-header" style="margin-top:16px;">Document — {score_to_pct(doc_score)}</div>', unsafe_allow_html=True)
                st.progress(doc_score)

                if result.get("extracted_pan_number"):
                    st.markdown(f"""
                    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5A6377;margin:12px 0 6px;">
                        Extracted Fields
                    </div>
                    <div style="background:#0F1117;border:1px solid #1A1E28;border-radius:2px;padding:12px;">
                        <div style="font-size:10px;color:#5A6377;letter-spacing:0.1em;margin-bottom:2px;">PAN NUMBER</div>
                        <div style="font-family:'DM Mono',monospace;font-size:14px;color:#3ECFCF;margin-bottom:10px;">{result.get("extracted_pan_number","--")}</div>
                        <div style="font-size:10px;color:#5A6377;letter-spacing:0.1em;margin-bottom:2px;">NAME</div>
                        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#C8CFD8;margin-bottom:10px;">{result.get("extracted_name","--")}</div>
                        <div style="font-size:10px;color:#5A6377;letter-spacing:0.1em;margin-bottom:2px;">DATE OF BIRTH</div>
                        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#C8CFD8;">{result.get("extracted_dob","--")}</div>
                    </div>""", unsafe_allow_html=True)

                rows = [
                    ("OCR Validation",      result.get("doc_ocr_valid", 0)),
                    ("Font Consistency",    result.get("doc_font_consistency", 0)),
                    ("Edge Consistency",    result.get("doc_edge_consistency", 0)),
                    ("Color Histogram",     result.get("doc_color_histogram", 0)),
                    ("Structural Align",    result.get("doc_structural_align", 0)),
                    ("Metadata Clean",      result.get("doc_metadata_clean", 0)),
                    ("SSIM Score",          result.get("doc_ssim_score", 0)),
                ]
                rows_html = "".join(render_score_row(l, v, i * 0.05) for i, (l, v) in enumerate(rows))
                st.markdown(f'<div style="margin-top:12px;">{rows_html}</div>', unsafe_allow_html=True)

            with col_b:
                bio_score = result.get("biometric_score", 0)
                st.markdown(f'<div class="section-header" style="margin-top:16px;">Biometric — {score_to_pct(bio_score)}</div>', unsafe_allow_html=True)
                st.progress(bio_score)

                bio_rows = [
                    ("Face Match",      result.get("face_match_score", 0)),
                    ("Liveness",        result.get("liveness_score", 0)),
                    ("Behavioral",      result.get("behavioral_score", 0)),
                ]
                bio_html = "".join(render_score_row(l, v, i * 0.07) for i, (l, v) in enumerate(bio_rows))
                st.markdown(f'<div style="margin-top:12px;">{bio_html}</div>', unsafe_allow_html=True)

                face_score = result.get("face_match_score", 0)
                live_score = result.get("liveness_score", 0)

                face_verdict = "MATCH" if face_score >= 0.80 else ("UNCERTAIN" if face_score >= 0.60 else "NO MATCH")
                live_verdict = "LIVE" if live_score >= 0.65 else ("UNCERTAIN" if live_score >= 0.45 else "SPOOF DETECTED")
                live_cls = "score-high" if live_score >= 0.65 else ("score-mid" if live_score >= 0.45 else "score-low")
                face_cls = "score-high" if face_score >= 0.80 else ("score-mid" if face_score >= 0.60 else "score-low")

                st.markdown(f"""
                <div style="margin-top:16px;">
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A6377;letter-spacing:0.1em;margin-bottom:10px;">VERDICTS</div>
                    <div style="display:flex;gap:10px;flex-wrap:wrap;">
                        <span class="score-value {face_cls}" style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.08em;">
                            FACE: {face_verdict}
                        </span>
                        <span class="score-value {live_cls}" style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.08em;">
                            LIVENESS: {live_verdict}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:12px;color:#2A2F3A;text-align:center;padding:60px 0;border:1px dashed #1A1E28;border-radius:2px;">Submit verification to view results</div>',
                unsafe_allow_html=True
            )

def render_aml_tab(use_mock):
    st.markdown('<div class="section-header">AML Transaction Monitoring</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)

        amount = st.number_input("Amount (INR)", min_value=1, max_value=10000000, value=75000, step=1000)
        txn_type = st.selectbox("Transaction Type", ["credit", "debit", "transfer_in", "transfer_out"])
        counterparty_country = st.selectbox("Counterparty Country", ["IN", "US", "GB", "IR", "KP", "SY", "PK", "AF", "AE", "SG"])
        account_age = st.number_input("Account Age (days)", min_value=0, max_value=3650, value=45)
        days_since_last = st.number_input("Days Since Last Transaction", min_value=0, max_value=365, value=2)

        st.markdown('<div class="section-header" style="margin-top:16px;">History Summary</div>', unsafe_allow_html=True)
        txn_count_24h = st.number_input("Transactions in Last 24h", min_value=0, max_value=50, value=2)
        unique_cps = st.number_input("Unique Counterparties (24h)", min_value=0, max_value=20, value=1)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        aml_btn = st.button("Analyze Transaction", key="run_aml")

    with col_right:
        st.markdown('<div class="section-header">Analysis Output</div>', unsafe_allow_html=True)

        if aml_btn:
            with st.spinner("Running AML pipeline..."):
                time.sleep(1.4)

            if use_mock:
                aml_result = mock_aml_result(amount)
            else:
                try:
                    payload = {
                        "transaction": {
                            "amount": amount,
                            "type": txn_type,
                            "timestamp": datetime.now().isoformat() + "Z",
                            "counterparty_country": counterparty_country,
                        },
                        "account_age_days": account_age,
                        "days_since_last_txn": days_since_last,
                    }
                    resp = requests.post(f"{BACKEND_URL}/aml/monitor/", json=payload, timeout=30)
                    aml_result = resp.json()
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    return

            st.session_state["aml_result"] = aml_result

        aml_result = st.session_state.get("aml_result")

        if aml_result:
            verdict = aml_result.get("verdict", "MONITOR")
            aml_score = aml_result.get("aml_score", 0)
            flags = aml_result.get("rule_flags", [])
            action = aml_result.get("recommended_action", "--")

            vmap = {"CLEAN": "approved", "SUSPICIOUS": "rejected", "MONITOR": "review"}
            vclass = vmap.get(verdict, "review")

            st.markdown(f"""
            <div class="verdict-card {vclass}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5A6377;">AML Verdict</div>
                        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#EAEEF4;margin-top:4px;">{verdict}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'DM Mono',monospace;font-size:32px;font-weight:500;color:#3ECFCF;">{score_to_pct(aml_score)}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.15em;color:#5A6377;text-transform:uppercase;">AML Score</div>
                    </div>
                </div>
                <div style="margin-top:14px;padding-top:12px;border-top:1px solid #1A1E28;display:flex;gap:20px;">
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.12em;color:#5A6377;text-transform:uppercase;">Action</div>
                        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#C8CFD8;margin-top:2px;">{action}</div>
                    </div>
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.12em;color:#5A6377;text-transform:uppercase;">Rules Triggered</div>
                        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#C8CFD8;margin-top:2px;">{len(flags)}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            sub_scores_html = "".join([
                render_score_row("Rule Engine",       aml_result.get("aml_score", 0), 0.0),
                render_score_row("Isolation Forest",  aml_result.get("isolation_forest_score", 0), 0.05),
                render_score_row("LSTM Sequential",   aml_result.get("lstm_anomaly_score", 0), 0.10),
            ])
            st.markdown(f'<div style="margin:12px 0;">{sub_scores_html}</div>', unsafe_allow_html=True)

            if flags:
                st.markdown('<div class="section-header" style="margin-top:16px;">Triggered Rules</div>', unsafe_allow_html=True)
                for i, flag in enumerate(flags):
                    sev = flag.get("severity", "LOW").lower()
                    code_cls = "rule-code" if sev == "high" else (f"rule-code {sev}")
                    delay_style = f"animation-delay:{i*0.08:.2f}s"
                    st.markdown(f"""
                    <div class="rule-flag {sev}" style="{delay_style}">
                        <div class="{code_cls}">{flag.get("code","--")}</div>
                        <div class="rule-detail">{flag.get("detail","No detail")} &nbsp;|&nbsp; Severity: {flag.get("severity","--")}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-bar">No compliance rules triggered for this transaction.</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:12px;color:#2A2F3A;text-align:center;padding:60px 0;border:1px dashed #1A1E28;border-radius:2px;">Submit a transaction to view analysis</div>',
                unsafe_allow_html=True
            )

def render_dashboard_tab():
    st.markdown('<div class="section-header">Case Review Dashboard</div>', unsafe_allow_html=True)

    stats_html = """
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
        <div class="dashboard-stat">
            <span class="stat-number">1,284</span>
            <span class="stat-label">Total Cases</span>
        </div>
        <div class="dashboard-stat">
            <span class="stat-number" style="color:#00D68F;">987</span>
            <span class="stat-label">Approved</span>
        </div>
        <div class="dashboard-stat">
            <span class="stat-number" style="color:#FFA502;">201</span>
            <span class="stat-label">Under Review</span>
        </div>
        <div class="dashboard-stat">
            <span class="stat-number" style="color:#FF4757;">96</span>
            <span class="stat-label">Rejected</span>
        </div>
    </div>"""
    st.markdown(stats_html, unsafe_allow_html=True)

    col_cases, col_aml = st.columns([1.4, 1], gap="large")

    with col_cases:
        st.markdown('<div class="section-header">Recent Verification Cases</div>', unsafe_allow_html=True)

        cases = [
            ("CASE-84721", "Priya Nair",        "0.91", "0.89", "approved"),
            ("CASE-84720", "Arjun Mehta",        "0.73", "0.77", "review"),
            ("CASE-84719", "Deepika Sharma",     "0.88", "0.92", "approved"),
            ("CASE-84718", "Rohit Kulkarni",     "0.42", "0.55", "rejected"),
            ("CASE-84717", "Sneha Iyer",         "0.95", "0.91", "approved"),
            ("CASE-84716", "Vikram Reddy",       "0.64", "0.69", "review"),
            ("CASE-84715", "Ananya Krishnan",    "0.87", "0.83", "approved"),
        ]

        header_html = """
        <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.15em;color:#2A2F3A;text-transform:uppercase;padding:6px 0;border-bottom:1px solid #1A1E28;margin-bottom:4px;">
            <span style="flex:1.2;">Case</span>
            <span style="flex:1.5;">Name</span>
            <span style="flex:0.7;text-align:right;">Doc</span>
            <span style="flex:0.7;text-align:right;">Bio</span>
            <span style="flex:0.8;text-align:right;">Status</span>
        </div>"""
        st.markdown(header_html, unsafe_allow_html=True)

        for i, (cid, name, doc, bio, status) in enumerate(cases):
            delay = f"{i*0.06:.2f}s"
            st.markdown(f"""
            <div class="case-row" style="animation-delay:{delay}">
                <span style="flex:1.2;color:#5A6377;">{cid}</span>
                <span style="flex:1.5;color:#C8CFD8;">{name}</span>
                <span style="flex:0.7;text-align:right;" class="{score_color_class(float(doc))}">{doc}</span>
                <span style="flex:0.7;text-align:right;" class="{score_color_class(float(bio))}">{bio}</span>
                <span style="flex:0.8;text-align:right;"><span class="status-pill status-{status}">{status.upper()}</span></span>
            </div>""", unsafe_allow_html=True)

    with col_aml:
        st.markdown('<div class="section-header">AML Alerts (Last 24h)</div>', unsafe_allow_html=True)

        alerts = [
            ("R1_LARGE_TRANSACTION", "HIGH",   "INR 85,000 transfer",       "14:22"),
            ("R2_STRUCTURING",       "HIGH",   "4 txns totalling INR 52k",  "12:45"),
            ("R7_RAPID_MOVEMENT",    "HIGH",   "Layering detected",          "11:30"),
            ("R9_MULTIPLE_CPS",      "MEDIUM", "11 counterparties in 24h",   "09:15"),
            ("R5_UNUSUAL_TIME",      "LOW",    "Transaction at 02:47",       "02:47"),
            ("R6_ROUND_NUMBER",      "LOW",    "INR 25,000 exact",           "01:15"),
        ]

        for i, (code, sev, detail, time_str) in enumerate(alerts):
            sev_l = sev.lower()
            code_cls = "rule-code" if sev_l == "high" else f"rule-code {sev_l}"
            delay = f"{i*0.07:.2f}s"
            st.markdown(f"""
            <div class="rule-flag {sev_l}" style="animation-delay:{delay};margin:5px 0;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div class="{code_cls}" style="font-size:10px;">{code}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#2A2F3A;">{time_str}</div>
                </div>
                <div class="rule-detail">{detail}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)

        dist_data = [
            ("Doc Score",      0.87, "score-high"),
            ("Biometric",      0.84, "score-high"),
            ("AML Score",      0.72, "score-high"),
            ("Avg Risk Total", 0.55, "score-mid"),
        ]

        for label, val, cls in dist_data:
            st.markdown(f"""
            <div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:10px;margin-bottom:4px;">
                    <span style="color:#5A6377;letter-spacing:0.05em;">{label}</span>
                    <span class="{cls}">{score_to_pct(val)}</span>
                </div>""", unsafe_allow_html=True)
            st.progress(val)

def main():
    backend, use_mock, doc_thresh, bio_thresh, aml_thresh = render_sidebar()

    st.markdown(
        '<div style="padding:8px 0 20px;">'
        '<div class="main-title">PAN Card Verification &amp; AML Monitoring System</div>'
        '<div class="main-subtitle">Know Your Customer  ·  Anti-Money Laundering  ·  RBI / FATF Compliance</div>'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["KYC Verification", "AML Monitoring", "Dashboard"])

    with tab1:
        render_kyc_tab(use_mock)

    with tab2:
        render_aml_tab(use_mock)

    with tab3:
        render_dashboard_tab()

if __name__ == "__main__":
    main()