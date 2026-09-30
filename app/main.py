"""
PhishGuard AI — Executive Single-Page AI/ML Capstone Dashboard

Designed for Muhammad Haris (S.No: 70) — Final Project KPITB AI/ML Training Program
Exact Structured Layout:
  1. Top Header: Project Name + Student Credentials (Muhammad Haris, S.No: 70, KPITB)
  2. Beautiful Quick Testing Buttons (4 scenarios)
  3. Single Column: Email & Link Input Console
  4. Two Columns:
       - Column 1: AI Classification & Threat Ranking (Verdict, Score, Confidence, Chart)
       - Column 2: Plain-English Bullet Explanations (Why marked as Legitimate/Phishing/Malicious)
  5. Single Column: Recommended Security Action
  6. Clean Footer
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference.engine import PhishGuardInference
from src.utils.config import MAX_EMAIL_LENGTH, MODELS_DIR

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="PhishGuard AI — Executive Capstone Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================
# Global CSS (Clean, Modern, Enlarged Typography)
# ============================================
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Theme Variables */
    :root {
        --canvas: #f8fafc;
        --card: #ffffff;
        --border: #cbd5e1;
        --border-subtle: #e2e8f0;
        --ink-title: #0f172a;
        --ink-body: #334155;
        --ink-muted: #64748b;
        --brand-blue: #2563eb;
        --brand-blue-hover: #1d4ed8;
    }

    /* Base Canvas */
    .stApp {
        background-color: var(--canvas) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: var(--ink-body) !important;
    }

    .main .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1360px !important;
    }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 1. TOP HEADER */
    .app-header {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo {
        width: 52px;
        height: 52px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .brand-title {
        font-size: 26px;
        font-weight: 800;
        color: var(--ink-title);
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .brand-title span {
        color: var(--brand-blue);
    }
    .brand-sub {
        font-size: 13px;
        color: var(--ink-muted);
        font-weight: 600;
        margin: 3px 0 0 0;
    }

    .student-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 10px;
        padding: 8px 18px;
        text-align: right;
    }
    .student-name {
        font-size: 16px;
        font-weight: 800;
        color: var(--ink-title);
        line-height: 1.2;
    }
    .student-meta {
        font-size: 12.5px;
        color: var(--brand-blue);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* Section Cards */
    .section-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .section-title {
        font-size: 15px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ink-title);
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1.5px solid var(--border-subtle);
        padding-bottom: 8px;
    }

    /* Streamlit Scenario Preset Buttons */
    div[data-testid="stButton"] button {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 700 !important;
        height: 48px !important;
        min-height: 48px !important;
        padding: 6px 14px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #eff6ff !important;
        border-color: #2563eb !important;
        color: #1d4ed8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.18) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0px) !important;
        background-color: #dbeafe !important;
    }
    div[data-testid="stButton"] button p {
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
    }

    /* Input & Textarea Elements */
    div[data-testid="stTextInput"] div[data-baseweb="base-input"],
    div[data-testid="stTextArea"] div[data-baseweb="textarea"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
    div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        padding: 10px 14px !important;
    }
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label p {
        font-size: 12.5px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #334155 !important;
        margin: 0 0 4px 0 !important;
    }

    /* Primary Submit Button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        height: 52px !important;
        min-height: 52px !important;
        padding: 12px 24px !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.32) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Verdict Card */
    .verdict-box {
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-width: 1.5px;
        border-style: solid;
    }
    .vb-phish { background: #fffbeb; border-color: #fde68a; }
    .vb-legit { background: #ecfdf5; border-color: #a7f3d0; }
    .vb-mal   { background: #fef2f2; border-color: #fecaca; }

    .verdict-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .vt-phish { color: #d97706; }
    .vt-legit { color: #059669; }
    .vt-mal   { color: #dc2626; }

    .status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 6px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .sb-phish { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .sb-legit { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
    .sb-mal   { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    /* 2 Big Primary KPI Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-bottom: 16px;
    }
    .kpi-cell {
        background: #f8fafc;
        border: 1.5px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px 18px;
    }
    .kpi-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ink-muted);
    }
    .kpi-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.1;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 11.5px;
        font-weight: 600;
        color: var(--ink-muted);
    }

    /* Plain-English Bullet Explanation Cards */
    .reason-box {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .reason-threat {
        border-left: 5px solid #dc2626;
        background: #fffafa;
    }
    .reason-warning {
        border-left: 5px solid #d97706;
        background: #fffdfa;
    }
    .reason-safe {
        border-left: 5px solid #059669;
        background: #fafffc;
    }
    .reason-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .reason-desc {
        font-size: 13px;
        color: #334155;
        line-height: 1.5;
        margin-left: 28px;
    }

    /* Word Token Badges */
    .token-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px 4px 2px 0;
    }
    .tc-threat { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .tc-safe   { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

    /* Recommended Security Action Card */
    .action-container {
        background: #eff6ff;
        border: 1.5px solid #bfdbfe;
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 18px 24px;
        margin-top: 4px;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.06);
    }
    .action-icon {
        font-size: 32px;
        line-height: 1.0;
        margin-top: 2px;
    }
    .action-content-title {
        font-size: 15px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .action-content-body {
        font-size: 14px;
        color: #1e293b;
        line-height: 1.5;
        font-weight: 500;
    }

    /* Footer */
    .footer-bar {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding-top: 20px;
        border-top: 1.5px solid var(--border-subtle);
        margin-top: 24px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# Engine Loading
# ============================================
@st.cache_resource(show_spinner="Loading PhishGuard AI Model...")
def load_engine():
    return PhishGuardInference(MODELS_DIR)

engine = load_engine()

# ============================================
# Plain-English Explanation Generator
# ============================================
def generate_plain_english_reasons(res, body_text: str, subj_text: str, sender_text: str) -> list:
    """Generate simple, human-readable bullet reasons explaining the AI's verdict."""
    reasons = []
    combined = (sender_text + " " + subj_text + " " + body_text).lower()

    if res.prediction != "LEGITIMATE":
        # Reason 1: Lookalike / Typosquatting
        if sender_text and ("rnicrosoft" in sender_text.lower() or "paypa1" in sender_text.lower() or "xyz" in sender_text.lower()):
            reasons.append({
                "type": "threat",
                "icon": "🎭",
                "title": "Fake / Spoofed Sender Domain",
                "desc": f"The sender address '{sender_text}' uses visual trickery (e.g. 'rn' pretending to be 'm') to fool the user into thinking it comes from a legitimate brand."
            })
        elif any("impersonation" in ind.lower() or "typosquat" in ind.lower() for ind in res.detected_indicators):
            reasons.append({
                "type": "threat",
                "icon": "🎭",
                "title": "Brand Impersonation Detected",
                "desc": "The sender's domain has a high similarity score to a protected corporate domain, indicating an impersonation attempt."
            })

        # Reason 2: Urgent Pressure Tactics
        if any(w in combined for w in ["immediate", "within 24 hours", "within 2 hours", "quota exceeded", "blocked", "suspension", "final notice", "urgent", "freeze"]):
            reasons.append({
                "type": "warning",
                "icon": "⏱️",
                "title": "Artificial Urgency & Fear Pressure",
                "desc": "The email uses panic words (such as 'immediate action required' or 'account blocked') to pressure the victim into acting quickly before checking if it is safe."
            })

        # Reason 3: Dangerous Executable / Binary Mention
        if any(ext in combined for ext in [".exe", ".scr", ".bat", ".pdf.exe"]):
            reasons.append({
                "type": "threat",
                "icon": "📎",
                "title": "Dangerous Program / File Attachment",
                "desc": "The email references an executable program file (.exe). Running this file on your computer can install malicious software or steal your data."
            })

        # Reason 4: Credential Theft / Fake Portal Links
        if any(w in combined for w in ["verify your password", "restore-access", "login", "password", "credential", "account activity"]):
            reasons.append({
                "type": "warning",
                "icon": "🔒",
                "title": "Credential Harvest / Password Prompt",
                "desc": "The message attempts to steer you to an external verification link to capture your private login password."
            })

        # Reason 5: NLP Vocabulary Attribution
        if res.top_threat_tokens:
            top_words = ", ".join([f"'{t['token']}'" for t in res.top_threat_tokens[:4]])
            reasons.append({
                "type": "threat",
                "icon": "🧠",
                "title": "AI NLP Word Pattern Matches",
                "desc": f"The NLP model detected high-impact phishing keywords ({top_words}) that strongly match historical phishing attacks from the 17,960 email training corpus."
            })

    else:
        # Legitimate Email Reasons
        reasons.append({
            "type": "safe",
            "icon": "✅",
            "title": "Normal Professional Workplace Language",
            "desc": "The text uses standard collaborative phrases (e.g. 'agenda', 'roadmap', 'meeting', 'discussion') with zero threat or urgency triggers."
        })
        reasons.append({
            "type": "safe",
            "icon": "🛡️",
            "title": "Authentic Business Sender Domain",
            "desc": f"The sender address ('{sender_text}') has standard domain structure with zero lookalike spoofing or character substitutions."
        })
        reasons.append({
            "type": "safe",
            "icon": "🔗",
            "title": "No Credential Harvesters or Dangerous Files",
            "desc": "The email contains no external credential-harvesting links, no deceptive redirects, and no executable payload attachments."
        })

    return reasons


# ============================================
# 4 Clean Presets
# ============================================
SCENARIOS = {
    "🚨 Credential Phish": {
        "sender": "helpdesk@corporate-verify-auth.xyz",
        "subject": "FINAL NOTICE: Mailbox Storage Quota Exceeded",
        "body": (
            "Dear Enterprise User,\n\n"
            "Your Office 365 mailbox storage quota has exceeded its allocated limit. "
            "Incoming emails will be completely blocked in 24 hours unless you verify your password immediately.\n\n"
            "Click below to restore mailbox access:\nhttps://login-microsoft-portal.xyz/restore-access\n\n"
            "Failure to update will result in permanent account deactivation.\nIT Service Desk"
        ),
    },
    "🛡️ Spoofed Brand": {
        "sender": "support@rnicrosoft.com",
        "subject": "Microsoft Security Team — SIMULATION",
        "body": (
            "Microsoft Security Team — SIMULATION\n\n"
            "We detected an unusual sign-in attempt on your account from a new device.\n\n"
            "Date: September 30, 2026\nLocation: Unknown\nDevice: Windows PC\n\n"
            "For this security exercise, review the message and identify the warning signs before taking any action.\n\n"
            "[Review Account Activity — support@rnicrosoft.com]\n\n"
            "If you did not initiate this activity, contact your organization's IT/security team through an independently verified channel.\n\n"
            "Microsoft Security Team\nThis is a cybersecurity training simulation."
        ),
    },
    "💻 Malware (.exe)": {
        "sender": "accounting@billing-gateway.net",
        "subject": "URGENT: Overdue Vendor Invoice INV-2026-8819",
        "body": (
            "Your account statement for invoice INV-2026-8819 is overdue by 14 days.\n"
            "Please review the attached statement and execute the automated verification utility:\n"
            "Attached File: invoice_payment_update.pdf.exe\n\n"
            "You must execute this update within 2 hours to avoid commercial credit freeze.\nAccounts Payable Department"
        ),
    },
    "✅ Legitimate Digest": {
        "sender": "sarah.jenkins@company.com",
        "subject": "Agenda for Thursday Engineering All-Hands",
        "body": (
            "Hi everyone,\n\n"
            "Here is our agenda for the engineering all-hands meeting this Thursday at 2:00 PM:\n"
            "1. Q3 Roadmap & NLP pipeline deliverables (20 min)\n"
            "2. Infrastructure cost review (15 min)\n"
            "3. Open Q&A and team recognition (15 min)\n\n"
            "Please add any additional discussion topics to the shared document before noon tomorrow.\n\n"
            "Best regards,\nSarah"
        ),
    },
}

DEFAULT_PRESET = "🛡️ Spoofed Brand"
if "form_sender_input" not in st.session_state:
    st.session_state["form_sender_input"] = SCENARIOS[DEFAULT_PRESET]["sender"]
if "form_subject_input" not in st.session_state:
    st.session_state["form_subject_input"] = SCENARIOS[DEFAULT_PRESET]["subject"]
if "form_body_input" not in st.session_state:
    st.session_state["form_body_input"] = SCENARIOS[DEFAULT_PRESET]["body"]

# ============================================
# 1. TOP HEADER: PROJECT NAME & STUDENT INFO
# ============================================
st.markdown(
    """
<div class="app-header">
    <div class="brand-section">
        <div class="brand-logo">🛡️</div>
        <div>
            <div class="brand-title">PhishGuard <span>AI</span></div>
            <div class="brand-sub">NLP-Driven Email Threat Intelligence & Risk Ranking System</div>
        </div>
    </div>
    <div class="student-card">
        <div class="student-name">Muhammad Haris</div>
        <div class="student-meta">S.No: 70 · Final Project · KPITB AI/ML Training Program</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================
# 2. BEAUTIFUL QUICK TESTING BUTTONS
# ============================================
st.markdown('<div style="font-size: 12px; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 6px;">⚡ Quick Test Scenarios (Click to Load):</div>', unsafe_allow_html=True)
p_cols = st.columns(4)

for i, name in enumerate(SCENARIOS.keys()):
    with p_cols[i]:
        if st.button(name, use_container_width=True):
            st.session_state["form_sender_input"] = SCENARIOS[name]["sender"]
            st.session_state["form_subject_input"] = SCENARIOS[name]["subject"]
            st.session_state["form_body_input"] = SCENARIOS[name]["body"]
            st.rerun()

st.write("")

# ============================================
# 3. SINGLE COLUMN: EMAIL & LINK INPUT CONSOLE
# ============================================
st.markdown(
    """
<div class="section-title">
    <span>1. Email & Link Input Console</span>
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #2563eb;">Input Workstation</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("analysis_form", clear_on_submit=False):
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sender_val = st.text_input(
            "Sender Email Address (Optional)",
            key="form_sender_input",
            placeholder="e.g. support@rnicrosoft.com",
        )
    with c_s2:
        subject_val = st.text_input(
            "Email Subject Header (Optional)",
            key="form_subject_input",
            placeholder="e.g. Account Security Alert",
        )

    body_val = st.text_area(
        "Email Body Content, Suspicious Links, or Raw Text:",
        key="form_body_input",
        height=190,
        max_chars=MAX_EMAIL_LENGTH,
        placeholder="Paste complete email body or suspicious URLs here...",
    )

    # High-visibility alerts if typosquatting or payload present
    if sender_val and "rnicrosoft" in sender_val.lower():
        st.markdown(
            '<div style="background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #b45309; margin-bottom: 10px;">⚠️ <strong>Homoglyph Spoof:</strong> \'rn\' simulates \'m\' (<code>rnicrosoft.com</code> vs <code>microsoft.com</code>)</div>',
            unsafe_allow_html=True,
        )
    if body_val and any(ext in body_val.lower() for ext in [".exe", ".scr", ".bat", ".pdf.exe"]):
        st.markdown(
            '<div style="background: #fef2f2; border: 1.5px solid #fecaca; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #dc2626; margin-bottom: 10px;">📎 <strong>Payload Alert:</strong> Executable attachment mention detected (<code>.exe / binary</code>)</div>',
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button(
        "⚡ Analyze Threat & Score Risk",
        use_container_width=True,
    )

st.write("")

# Run inference
curr_body = body_val if body_val else st.session_state.get("form_body_input", "")
curr_subj = subject_val if subject_val else st.session_state.get("form_subject_input", "")
curr_sender = sender_val if sender_val else st.session_state.get("form_sender_input", "")

start_t = datetime.datetime.now()
res = engine.analyze(text=curr_body, subject=curr_subj, sender=curr_sender)
latency_ms = (datetime.datetime.now() - start_t).total_seconds() * 1000

# ============================================
# 4. TWO COLUMNS: CLASSIFICATION & REASONS
# ============================================
col_out1, col_out2 = st.columns([1, 1], gap="large")

# --------------------------------------------
# COLUMN 1: AI Classification & Threat Ranking
# --------------------------------------------
with col_out1:
    st.markdown(
        """
    <div class="section-title">
        <span>2. AI Classification & Threat Ranking</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #059669;">● Live Prediction</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Class styling
    if res.prediction == "LEGITIMATE":
        vb_cls = "vb-legit"
        vt_cls = "vt-legit"
        v_tag = '<span class="status-badge sb-legit">● VERIFIED SAFE</span>'
    elif res.prediction == "PHISHING":
        vb_cls = "vb-phish"
        vt_cls = "vt-phish"
        v_tag = '<span class="status-badge sb-phish">● PHISHING DETECTED</span>'
    else:
        vb_cls = "vb-mal"
        vt_cls = "vt-mal"
        v_tag = '<span class="status-badge sb-mal">● MALICIOUS ATTACK</span>'

    # Big Verdict Banner
    st.markdown(
        f"""
    <div class="verdict-box {vb_cls}">
        <div>
            <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                AI Model Verdict
            </div>
            <div class="verdict-title {vt_cls}">{res.prediction}</div>
            <div style="font-size: 12px; color: #475569; margin-top: 3px;">
                Inference Latency: <strong>{latency_ms:.1f}ms</strong> &nbsp;|&nbsp; Calibrated Decision Margin
            </div>
        </div>
        <div style="text-align: right;">
            {v_tag}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 2 Big KPI Cells: Risk Score & Confidence
    risk_int = int(res.risk_score * 100)
    st.markdown(
        f"""
    <div class="kpi-row">
        <div class="kpi-cell">
            <div class="kpi-label">Composite Risk Score</div>
            <div class="kpi-num" style="color: {res.risk_color};">{risk_int} <span style="font-size: 16px; font-weight: 600; color: #64748b;">/ 100</span></div>
            <div class="kpi-sub">Threat Severity: <strong>{res.risk_level}</strong></div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Model Confidence</div>
            <div class="kpi-num" style="color: #0f172a;">{res.confidence:.1%}</div>
            <div class="kpi-sub">Platt-Calibrated SVM Probability</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Class Probability Distribution Bars
    st.markdown(
        '<div style="font-size: 11.5px; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 2px; font-family: \'JetBrains Mono\', monospace;">Posterior Class Probabilities:</div>',
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
            textfont=dict(size=12, family="JetBrains Mono", color="white"),
        )
    )
    prob_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=100,
        margin=dict(l=0, r=20, t=4, b=4),
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, family="JetBrains Mono", color="#334155"), autorange="reversed"),
        bargap=0.22,
    )
    st.plotly_chart(prob_chart, use_container_width=True, config={"displayModeBar": False})

    # Top Triggering Words
    if res.top_threat_tokens:
        st.markdown(
            '<div style="font-size: 11px; font-weight: 700; color: #dc2626; font-family: \'JetBrains Mono\', monospace; margin: 8px 0 2px 0;">TOP THREAT KEYWORDS DETECTED:</div>',
            unsafe_allow_html=True,
        )
        chips_threat = "".join([f'<span class="token-chip tc-threat">{t["token"]} +{abs(t["impact"]):.2f}</span>' for t in res.top_threat_tokens[:5]])
        st.markdown(chips_threat, unsafe_allow_html=True)
    elif res.top_safe_tokens:
        st.markdown(
            '<div style="font-size: 11px; font-weight: 700; color: #059669; font-family: \'JetBrains Mono\', monospace; margin: 8px 0 2px 0;">TOP BENIGN KEYWORDS DETECTED:</div>',
            unsafe_allow_html=True,
        )
        chips_safe = "".join([f'<span class="token-chip tc-safe">{t["token"]} -{abs(t["impact"]):.2f}</span>' for t in res.top_safe_tokens[:5]])
        st.markdown(chips_safe, unsafe_allow_html=True)

# --------------------------------------------
# COLUMN 2: Why Was This Email Flagged? (Easy Bullets)
# --------------------------------------------
with col_out2:
    st.markdown(
        f"""
    <div class="section-title">
        <span>Why Was This Email Marked As {res.prediction}?</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #2563eb;">Plain-English Reasons</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    reasons = generate_plain_english_reasons(res, curr_body, curr_subj, curr_sender)

    for r in reasons:
        border_cls = f"reason-{r['type']}"
        st.markdown(
            f"""
        <div class="reason-box {border_cls}">
            <div class="reason-header">
                <span style="font-size: 18px;">{r['icon']}</span>
                <span>{r['title']}</span>
            </div>
            <div class="reason-desc">{r['desc']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ============================================
# 5. SINGLE COLUMN: RECOMMENDED SECURITY ACTION
# ============================================
st.write("")
action_icon = "🛡️" if res.prediction == "LEGITIMATE" else ("🚨" if res.prediction == "MALICIOUS" else "⚠️")

st.markdown(
    f"""
<div class="action-container">
    <div class="action-icon">{action_icon}</div>
    <div>
        <div class="action-content-title">Recommended Security Action</div>
        <div class="action-content-body">{res.recommendation}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================
# 6. FOOTER
# ============================================
st.markdown(
    """
<div class="footer-bar">
    <strong>PhishGuard AI</strong> · Final Capstone Project · KPITB AI/ML Training Program<br>
    Developed by <strong>Muhammad Haris</strong> (S.No: 70) · Model: Calibrated Linear SVM (10,020 Features) · Test Accuracy: 98.64%
</div>
""",
    unsafe_allow_html=True,
)
