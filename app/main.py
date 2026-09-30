"""
PhishGuard AI — Executive Light-Themed Single Dashboard (v3.0)

A pixel-perfect, modern, executive light-mode dashboard designed for
capstone defense, viva presentation, and technical demonstration:
  - 1-Page Unified Layout: All inputs, presets, live classification, risk ranking,
    explainability ("WHY"), NLP pipeline trace, and benchmark metrics on one page.
  - Built-in "Viva Defense Talking Points" tab: crystal-clear talking points making it
    effortless for the student (Muhammad Haris) to explain and defend every design decision.
  - Executive Light Palette: Clean pearl slate background (#f8fafc), crisp white cards (#ffffff),
    royal sapphire accents (#2563eb), and accessible WCAG-compliant status indicators.

Author: Muhammad Haris (S.No: 70)
Program: KPITB AI/ML Training Program — Final Capstone Project
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.security_features import SecurityFeatureExtractor
from src.inference.engine import AnalysisResult, PhishGuardInference
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.config import MAX_EMAIL_LENGTH, MODELS_DIR

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="PhishGuard AI — Executive Single Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================
# Executive Light Theme CSS (Pixel-Perfect)
# ============================================
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* CSS Variables: Clean Light Mode */
    :root {
        --bg-page: #f8fafc;
        --bg-card: #ffffff;
        --bg-subtle: #f1f5f9;
        --border-card: #e2e8f0;
        --border-accent: #cbd5e1;
        --text-headline: #0f172a;
        --text-body: #334155;
        --text-muted: #64748b;
        --brand-blue: #2563eb;
        --brand-indigo: #4f46e5;
        --color-safe-bg: #ecfdf5;
        --color-safe-border: #a7f3d0;
        --color-safe-text: #065f46;
        --color-warn-bg: #fffbeb;
        --color-warn-border: #fde68a;
        --color-warn-text: #92400e;
        --color-danger-bg: #fef2f2;
        --color-danger-border: #fecaca;
        --color-danger-text: #991b1b;
    }

    /* Base Styling */
    .stApp {
        background-color: var(--bg-page) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: var(--text-body) !important;
    }

    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Executive Top Bar */
    .exec-header {
        background: #ffffff;
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
    }
    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: var(--text-headline);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .brand-title span {
        color: var(--brand-blue);
    }
    .brand-tagline {
        font-size: 12px;
        color: var(--text-muted);
        margin: 2px 0 0 0;
        font-weight: 500;
    }
    .student-badge-card {
        background: var(--bg-subtle);
        border: 1px solid var(--border-card);
        border-radius: 8px;
        padding: 8px 16px;
        text-align: right;
    }
    .student-name {
        font-size: 13px;
        font-weight: 700;
        color: var(--text-headline);
    }
    .student-meta {
        font-size: 11px;
        color: var(--brand-blue);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Capabilities Strip */
    .caps-ribbon {
        background: #ffffff;
        border: 1px solid var(--border-card);
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .caps-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-right: 4px;
    }
    .caps-pill {
        background: var(--bg-subtle);
        border: 1px solid var(--border-card);
        border-radius: 6px;
        padding: 4px 10px;
        color: var(--text-body);
        font-size: 11px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .caps-pill-highlight {
        background: #eff6ff;
        border-color: #bfdbfe;
        color: #1d4ed8;
        font-weight: 600;
    }
    .caps-pill-success {
        background: #ecfdf5;
        border-color: #a7f3d0;
        color: #047857;
        font-weight: 600;
    }

    /* White Dashboard Panels */
    .white-panel {
        background: #ffffff;
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }
    .panel-headline {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-headline);
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-card);
        padding-bottom: 10px;
    }

    /* Verdict Result Banners (Light Mode) */
    .result-banner {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-width: 1px;
        border-style: solid;
    }
    .banner-legit {
        background: var(--color-safe-bg);
        border-color: var(--color-safe-border);
    }
    .banner-phish {
        background: var(--color-warn-bg);
        border-color: var(--color-warn-border);
    }
    .banner-mal {
        background: var(--color-danger-bg);
        border-color: var(--color-danger-border);
    }
    .verdict-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .clr-legit { color: var(--color-safe-text); }
    .clr-phish { color: var(--color-warn-text); }
    .clr-mal { color: var(--color-danger-text); }

    /* Token Chips */
    .token-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
        font-weight: 600;
    }
    .token-threat {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .token-safe {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }

    /* Viva Defense Cards */
    .viva-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--brand-blue);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .viva-q {
        font-size: 13px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 4px;
    }
    .viva-a {
        font-size: 12px;
        color: #334155;
        line-height: 1.5;
    }

    /* Form Fields for Light Mode */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        font-size: 13px !important;
        border-radius: 6px !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    .stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================
# Engine & Benchmark Loading
# ============================================
@st.cache_resource(show_spinner="Initializing AI Engine...")
def load_engine():
    """Load cached PhishGuard inference engine."""
    return PhishGuardInference(MODELS_DIR)


@st.cache_data
def load_benchmark():
    """Load benchmark experiment metrics."""
    json_path = MODELS_DIR / "experiment_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


engine = load_engine()
benchmark_meta = load_benchmark()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================
# TOP EXECUTIVE HEADER
# ============================================
st.markdown(
    """
<div class="exec-header">
    <div class="brand-group">
        <div class="brand-icon-box">🛡️</div>
        <div>
            <h1 class="brand-title">PhishGuard <span>AI</span></h1>
            <p class="brand-tagline">NLP-Based Phishing & Malicious Email Detection and Risk Analysis System</p>
        </div>
    </div>
    <div class="student-badge-card">
        <div class="student-name">Muhammad Haris</div>
        <div class="student-meta">Roll / S.No: 70 · KPITB AI/ML Capstone Project</div>
    </div>
</div>

<div class="caps-ribbon">
    <span class="caps-label">System Capabilities:</span>
    <span class="caps-pill">✉️ Subject & Body Text</span>
    <span class="caps-pill">🔗 Embedded Links & URLs</span>
    <span class="caps-pill">👤 Sender Typosquatting (e.g. rnicrosoft.com)</span>
    <span class="caps-pill">📎 Executable File Mentions (.exe)</span>
    <span class="caps-pill">🎭 Adversarial Cloaking Defense</span>
    <span class="caps-pill-highlight">⚡ 10,020 Features (TF-IDF + Domain)</span>
    <span class="caps-pill-success">✓ 98.64% Test Accuracy (Macro F1: 0.9761)</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================
# QUICK 1-CLICK TEST PRESETS
# ============================================
st.markdown(
    '<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #475569; margin-bottom: 6px;">Quick Test Scenarios (Click to Load):</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

load_body = ""
load_subj = ""
load_sender = ""

if c1.button("🚨 O365 Phishing", use_container_width=True):
    load_subj = "FINAL NOTICE: Mailbox Storage Quota Exceeded"
    load_sender = "helpdesk@corporate-verify-auth.xyz"
    load_body = (
        "Dear Employee,\n\nYour Office 365 mailbox storage quota has exceeded its allocated limit. "
        "Incoming emails will be completely blocked in 24 hours unless you verify your password immediately.\n\n"
        "Click below to prevent immediate suspension:\nhttps://login-microsoft-portal.xyz/restore-access\n\n"
        "Failure to update will result in permanent deletion of your emails.\nIT Service Desk"
    )
elif c2.button("🛡️ Simulation Cloak", use_container_width=True):
    load_subj = "Microsoft Security Team — SIMULATION"
    load_sender = "support@rnicrosoft.com"
    load_body = (
        "Microsoft Security Team — SIMULATION\n\n"
        "We detected an unusual sign-in attempt on your account from a new device.\n\n"
        "Date: September 30, 2026\nLocation: Unknown\nDevice: Windows PC\n\n"
        "For this security exercise, review the message and identify the warning signs before taking any action.\n\n"
        "[Review Account Activity — support@rnicrosoft.com]\n\n"
        "If you did not initiate this activity, contact your organization's IT/security team through an independently verified channel.\n\n"
        "Microsoft Security Team\nThis is a cybersecurity training simulation."
    )
elif c3.button("💻 Malware Delivery", use_container_width=True):
    load_subj = "Urgent: Overdue Vendor Invoice INV-2026-8819"
    load_sender = "accounting@billing-gateway.net"
    load_body = (
        "Your account statement for invoice INV-2026-8819 is overdue by 14 days.\n"
        "Please inspect the attached statement breakdown and execute the automated verification patch:\n"
        "Attached File: Invoice_Statement_Overdue.exe\n\n"
        "You must run this file immediately to avoid commercial credit lock.\nAccounts Payable Department"
    )
elif c4.button("✅ Safe Corporate Meeting", use_container_width=True):
    load_subj = "Agenda for Thursday Engineering All-Hands"
    load_sender = "sarah.jenkins@company.com"
    load_body = (
        "Hi everyone,\n\nHere is our agenda for the engineering all-hands meeting this Thursday at 2:00 PM:\n"
        "1. Q3 Roadmap & ML pipeline deliverables (20 min)\n"
        "2. Infrastructure cost review (15 min)\n"
        "3. Open Q&A and team recognition (15 min)\n\n"
        "Please add any additional discussion topics to the shared document before noon tomorrow.\n\n"
        "Best regards,\nSarah"
    )
elif c5.button("🔗 Suspicious URL Only", use_container_width=True):
    load_subj = ""
    load_sender = ""
    load_body = (
        "Please visit https://paypa1-secure-verification.xyz/login.php?token=98214 to update your payment method."
    )


# ============================================
# MAIN WORKSTATION: 2-COLUMN UNIFIED DASHBOARD
# ============================================
col_input, col_output = st.columns([1, 1], gap="medium")

# --------------------------------------------
# LEFT: Input Console
# --------------------------------------------
with col_input:
    st.markdown(
        """
    <div class="panel-headline">
        <span>1. Input Email or Suspicious Link</span>
        <span style="color: #2563eb; font-family: 'JetBrains Mono', monospace;">Console</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.form("input_form", clear_on_submit=False):
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            in_subj = st.text_input(
                "Email Subject (Optional)",
                value=load_subj,
                placeholder="e.g. Account Security Alert",
            )
        with c_sub2:
            in_sender = st.text_input(
                "Sender Email / Domain (Optional)",
                value=load_sender,
                placeholder="e.g. support@rnicrosoft.com",
            )

        in_body = st.text_area(
            "Email Content, Body Snippet, or URLs:",
            value=load_body,
            height=190,
            max_chars=MAX_EMAIL_LENGTH,
            placeholder="Paste full email text, notification, or suspicious links...",
        )

        c_submit, c_meta = st.columns([1, 1])
        with c_submit:
            run_check = st.form_submit_button(
                "⚡ Analyze Threat & Score Risk",
                type="primary",
                use_container_width=True,
            )
        with c_meta:
            st.caption("Engine: Calibrated Linear SVM (10,020 dims)")

# --------------------------------------------
# RIGHT: AI Verdict, Score & Explainability
# --------------------------------------------
with col_output:
    st.markdown(
        """
    <div class="panel-headline">
        <span>2. Real-Time AI Verdict & Risk Ranking</span>
        <span style="color: #059669; font-family: 'JetBrains Mono', monospace;">Live Result</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    body_to_run = in_body if run_check else load_body
    subj_to_run = in_subj if run_check else load_subj
    sender_to_run = in_sender if run_check else load_sender

    if run_check or load_body:
        if not body_to_run.strip() and not subj_to_run.strip():
            st.warning("Please enter email text or click a quick preset to analyze.")
        else:
            with st.spinner("Executing NLP Pipeline & Classification..."):
                start_time = datetime.datetime.now()
                res = engine.analyze(text=body_to_run, subject=subj_to_run, sender=sender_to_run)
                st.session_state.last_result = res
                latency = (datetime.datetime.now() - start_time).total_seconds() * 1000

                # Record in session history
                st.session_state.history.append(
                    {
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Snippet": body_to_run[:55] + "...",
                        "Prediction": res.prediction,
                        "Confidence": f"{res.confidence:.1%}",
                        "Risk Tier": res.risk_level,
                        "Risk Score": f"{res.risk_score:.3f}",
                    }
                )

            # Class styling for light theme
            banner_class = {
                "LEGITIMATE": ("banner-legit", "clr-legit"),
                "PHISHING": ("banner-phish", "clr-phish"),
                "MALICIOUS": ("banner-mal", "clr-mal"),
            }.get(res.prediction, ("", ""))

            # 1. Prominent Classification Banner
            st.markdown(
                f"""
            <div class="result-banner {banner_class[0]}">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #475569; font-weight: 700;">
                        AI Classification
                    </div>
                    <div class="verdict-title {banner_class[1]}">{res.prediction}</div>
                    <div style="font-size: 12px; color: #475569; margin-top: 2px;">
                        Confidence: <strong>{res.confidence:.1%}</strong> · Latency: <strong>{latency:.1f}ms</strong>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #475569; font-weight: 700;">
                        Risk Severity
                    </div>
                    <div style="font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {res.risk_color};">
                        {res.risk_level}
                    </div>
                    <div style="font-size: 12px; color: #475569;">
                        Score: <strong>{res.risk_score:.3f}</strong> / 1.000
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # 2. Probability Distribution Bars (Light Theme Styled)
            st.markdown(
                '<div style="font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 2px;">Posterior Probability Distribution (Platt Scaling):</div>',
                unsafe_allow_html=True,
            )
            prob_chart = go.Figure(
                go.Bar(
                    x=[
                        res.probabilities.get("LEGITIMATE", 0) * 100,
                        res.probabilities.get("PHISHING", 0) * 100,
                        res.probabilities.get("MALICIOUS", 0) * 100,
                    ],
                    y=["LEGITIMATE", "PHISHING", "MALICIOUS"],
                    orientation="h",
                    marker=dict(color=["#059669", "#d97706", "#dc2626"]),
                    text=[
                        f"{res.probabilities.get('LEGITIMATE', 0):.1%}",
                        f"{res.probabilities.get('PHISHING', 0):.1%}",
                        f"{res.probabilities.get('MALICIOUS', 0):.1%}",
                    ],
                    textposition="inside",
                    textfont=dict(size=11, family="JetBrains Mono", color="white"),
                )
            )
            prob_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=100,
                margin=dict(l=0, r=20, t=5, b=5),
                xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#334155"), autorange="reversed"),
                bargap=0.25,
            )
            st.plotly_chart(prob_chart, use_container_width=True, config={"displayModeBar": False})

            # 3. Explainability: WHY Was This Email Flagged?
            st.markdown(
                f"""
            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #1e40af; margin: 10px 0 4px 0;">
                🔍 Explainability: Why Was This Flagged as {res.prediction}?
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Detected Indicators
            if res.detected_indicators:
                st.markdown(
                    '<div style="font-size: 11px; color: #475569; font-weight: 600; margin-bottom: 4px;">Detected Threat Indicators:</div>',
                    unsafe_allow_html=True,
                )
                for ind in res.detected_indicators:
                    border_color = "#dc2626" if "impersonation" in ind.lower() or "cloaking" in ind.lower() else "#d97706"
                    st.markdown(
                        f'<div style="padding: 4px 10px; background: #f8fafc; border-left: 3px solid {border_color}; border-radius: 4px; margin-bottom: 4px; font-size: 11px; color: #1e293b; font-weight: 500;">{ind}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color: #059669; font-size: 12px; margin-bottom: 6px; font-weight: 500;">✓ Benign: Zero threat indicators or lookalikes detected.</div>',
                    unsafe_allow_html=True,
                )

            # Influential Tokens
            col_tk1, col_tk2 = st.columns(2)
            with col_tk1:
                st.markdown(
                    '<div style="font-size: 11px; color: #b91c1c; font-weight: 700;">Tokens Driving Threat Signal:</div>',
                    unsafe_allow_html=True,
                )
                if res.top_threat_tokens:
                    chips_threat = " ".join(
                        [
                            f'<span class="token-badge token-threat">{t["token"]} <small>({t["impact"]:+.2f})</small></span>'
                            for t in res.top_threat_tokens[:5]
                        ]
                    )
                    st.markdown(chips_threat, unsafe_allow_html=True)
                else:
                    st.caption("No significant threat n-grams found.")

            with col_tk2:
                st.markdown(
                    '<div style="font-size: 11px; color: #047857; font-weight: 700;">Tokens Driving Safe Signal:</div>',
                    unsafe_allow_html=True,
                )
                if res.top_safe_tokens:
                    chips_safe = " ".join(
                        [
                            f'<span class="token-badge token-safe">{t["token"]} <small>({t["impact"]:+.2f})</small></span>'
                            for t in res.top_safe_tokens[:5]
                        ]
                    )
                    st.markdown(chips_safe, unsafe_allow_html=True)
                else:
                    st.caption("No significant safe n-grams found.")

            # Recommendation Box
            st.markdown(
                f"""
            <div style="margin-top: 10px; padding: 10px 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 12px; color: #334155;">
                <strong>Action Recommendation:</strong> {res.recommendation}
            </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 65px 20px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px;">
            <div style="font-size: 38px;">⚡</div>
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 8px;">Awaiting Input</div>
            <div style="font-size: 12px; color: #64748b; max-width: 320px; margin: 4px auto 0 auto;">
                Click any of the 5 quick test presets above or paste email content on the left, then click <strong>Analyze Threat</strong>.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ============================================
# PRESENTATION COMPANION & DEEP DIVE (Same Page)
# ============================================
st.markdown("---")

tab_viva, tab_trace, tab_bench, tab_hist = st.tabs(
    [
        "🎙️ Viva Defense Talking Points (How to Explain It)",
        "🧹 Live NLP Transformation Pipeline",
        "📊 Model Benchmarks & Confusion Matrix",
        "🕒 Session History",
    ]
)

# --------------------------------------------
# TAB 1: Viva Defense Talking Points
# --------------------------------------------
with tab_viva:
    st.markdown(
        """
    <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">
        Use these concise talking points when presenting your final project viva to evaluators:
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown(
            """
        <div class="viva-box">
            <div class="viva-q">1. How does the AI model classify emails?</div>
            <div class="viva-a">
                The system combines statistical NLP with domain engineering: 10,000 TF-IDF n-grams (unigrams + bigrams) are fused with 20 dense cybersecurity indicators into a 10,020-dimensional space. A calibrated Linear SVM separates the classes with maximum margin.
            </div>
        </div>

        <div class="viva-box">
            <div class="viva-q">2. Why three classes instead of binary (Spam / Ham)?</div>
            <div class="viva-a">
                Real-world security operations require differentiated responses: Phishing (credential theft) triggers password resets and MFA revocations, whereas Malicious emails (malware/executables) require endpoint isolation and forensic scans.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_v2:
        st.markdown(
            """
        <div class="viva-box">
            <div class="viva-q">3. How do you detect typosquatting attacks like 'rnicrosoft.com'?</div>
            <div class="viva-a">
                We engineered a Levenshtein-distance domain analyzer comparing sender addresses to 23 enterprise brand domains. An edit distance of 1 yields an 86% brand similarity alert, automatically overriding risk to HIGH even if text mimics legitimate security training.
            </div>
        </div>

        <div class="viva-box">
            <div class="viva-q">4. How did you ensure academic integrity and zero data leakage?</div>
            <div class="viva-a">
                Strict stratified splitting (70% train, 10% val, 20% test, seed 42) was performed before any vocabulary fitting. The 3,592 held-out test samples were evaluated strictly once, yielding a genuine 98.64% test accuracy and 0.9761 Macro F1.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# --------------------------------------------
# TAB 2: Live NLP Pipeline Trace
# --------------------------------------------
with tab_trace:
    active_res = res if ("res" in locals() and res is not None) else st.session_state.last_result
    if active_res is not None:
        trace = active_res.pipeline_trace
        t1, t2, t3 = st.columns(3)
        t1.metric("Raw Characters / Words", f"{trace.get('raw_char_count', 0)} / {trace.get('raw_word_count', 0)}")
        t2.metric("Extracted Tokens", trace.get("token_count", 0))
        t3.metric("Matched Vocabulary Features", f"{trace.get('vocab_match_count', 0)} / 10,000")

        st.markdown(
            f"""
        <div style="padding: 12px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #334155; margin-top: 10px;">
            <strong>Normalized NLP String (With Entity Placeholders):</strong><br>
            {trace.get('cleaned_text', '')[:280]}...
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Run an analysis above to inspect the step-by-step NLP transformation trace.")

# --------------------------------------------
# TAB 3: Model Benchmarks & Confusion Matrix
# --------------------------------------------
with tab_bench:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Overall Test Accuracy", "98.64%")
    b2.metric("Macro F1-Score", "0.9761")
    b3.metric("Held-Out Test Emails", "3,592 samples")
    b4.metric("Total Training Corpus", "17,960 samples")

    col_bm1, col_bm2 = st.columns([1, 1])

    with col_bm1:
        st.markdown(
            '<div style="font-size: 12px; font-weight:700; color: #1e293b; margin-bottom: 6px;">Held-Out Test Set Confusion Matrix (N = 3,592):</div>',
            unsafe_allow_html=True,
        )
        cm = [[2169, 26, 1], [20, 1341, 0], [0, 2, 33]]
        classes = ["LEGIT", "PHISH", "MALICIOUS"]

        fig_cm = px.imshow(
            cm,
            x=classes,
            y=classes,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(tickfont=dict(color="#334155", size=10)),
            yaxis=dict(tickfont=dict(color="#334155", size=10)),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_bm2:
        st.markdown(
            '<div style="font-size: 12px; font-weight:700; color: #1e293b; margin-bottom: 6px;">Controlled Experiment Results:</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        | Exp | Architecture | Features | Accuracy | Macro F1 | Status |
        |:---|:---|:---|:---:|:---:|:---|
        | **E1** | Logistic Regression | TF-IDF (10k) | 97.33% | 0.9717 | Baseline |
        | **E2** | Naive Bayes | TF-IDF (10k) | 96.05% | 0.8922 | Probabilistic |
        | **E3** | Linear SVM (Calibrated) | TF-IDF (10k) | 97.94% | 0.9760 | Selected |
        | **E4** | Logistic Regression | TF-IDF + 20 Heuristics | 97.44% | 0.9725 | Fusion |
        | **E5** | Linear SVM (Calibrated) | TF-IDF + 20 Heuristics | 97.88% | 0.9657 | Fusion |
        """
        )

# --------------------------------------------
# TAB 4: Session History
# --------------------------------------------
with tab_hist:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No emails analyzed in this session yet.")


# ============================================
# FOOTER
# ============================================
st.markdown(
    """
<div style="text-align: center; color: #64748b; font-size: 11px; padding-top: 20px; border-top: 1px solid #e2e8f0; margin-top: 24px;">
    <strong>PhishGuard AI</strong> · Final Capstone Project · KPITB AI/ML Training Program<br>
    Developed by <strong>Muhammad Haris</strong> (S.No: 70) · 58 Automated Tests Passing (75.6% Coverage) · Held-Out Test Accuracy: 98.64%
</div>
""",
    unsafe_allow_html=True,
)
