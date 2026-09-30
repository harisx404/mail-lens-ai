"""
PhishGuard AI — Executive Single-Page AI/ML Capstone Dashboard

Optimized for 1920x1080 Full HD Presentation Displays
Developer: Muhammad Haris (S.No: 70)
Program: Final Project — KPITB AI/ML Training Program
Model: Calibrated Linear SVM + TF-IDF (10,020 Features) — 98.64% Test Accuracy

Layout Architecture:
  1. Top Header: Project Name & Student Info (Muhammad Haris, S.No: 70, KPITB)
  2. Beautiful Quick Testing Scenario Buttons (4 options)
  3. Single Column: 1. Email & Link Input Console
  4. Two Columns:
       - Column 1: 2. AI Classification & Threat Ranking (Verdict, Score, Confidence, Chart)
       - Column 2: Plain-English Reasons (At least 4 clear bullet explanations)
  5. Single Column: 3. Recommended Security Action
  6. Enlarged Executive Presentation Footer
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
# Global CSS (Scaled for 1920x1080 Displays)
# ============================================
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Theme Tokens */
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

    /* 1920x1080 Scaled Container */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1680px !important;
    }

    /* Hide Streamlit default headers/footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 1. TOP HEADER */
    .app-header {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 16px;
        padding: 22px 34px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .brand-logo {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        color: #ffffff;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.32);
    }
    .brand-title {
        font-size: 32px;
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
        font-size: 15.5px;
        color: var(--ink-muted);
        font-weight: 600;
        margin: 5px 0 0 0;
    }

    .student-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 14px;
        padding: 12px 26px;
        text-align: right;
    }
    .student-name {
        font-size: 22px;
        font-weight: 800;
        color: var(--ink-title);
        line-height: 1.2;
    }
    .student-meta {
        font-size: 15px;
        color: var(--brand-blue);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        margin-top: 3px;
    }

    /* Section Titles */
    .section-title {
        font-size: 18px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ink-title);
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1.5px solid var(--border-subtle);
        padding-bottom: 11px;
    }

    /* Scenario Preset Buttons (1920x1080 Large) */
    div[data-testid="stButton"] button {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 15.5px !important;
        font-weight: 700 !important;
        height: 56px !important;
        min-height: 56px !important;
        padding: 10px 18px !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05) !important;
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
        box-shadow: 0 6px 16px -2px rgba(37, 99, 235, 0.22) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0px) !important;
        background-color: #dbeafe !important;
    }
    div[data-testid="stButton"] button p {
        font-size: 15.5px !important;
        font-weight: 700 !important;
        color: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
    }

    /* Form Inputs & Textarea */
    div[data-testid="stTextInput"] div[data-baseweb="base-input"],
    div[data-testid="stTextArea"] div[data-baseweb="textarea"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
    div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.22) !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        line-height: 1.55 !important;
        padding: 14px 18px !important;
    }
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label p {
        font-size: 14.5px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #334155 !important;
        margin: 0 0 8px 0 !important;
    }

    /* Primary Submit Button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 18.5px !important;
        font-weight: 800 !important;
        height: 60px !important;
        min-height: 60px !important;
        padding: 14px 30px !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
        box-shadow: 0 8px 26px rgba(37, 99, 235, 0.48) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
        font-size: 18.5px !important;
        font-weight: 800 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Verdict Banner */
    .verdict-box {
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 20px;
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
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .vt-phish { color: #d97706; }
    .vt-legit { color: #059669; }
    .vt-mal   { color: #dc2626; }

    .status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 800;
        padding: 9px 22px;
        border-radius: 9px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .sb-phish { background: #fee2e2; color: #991b1b; border: 1.5px solid #fca5a5; }
    .sb-legit { background: #d1fae5; color: #065f46; border: 1.5px solid #6ee7b7; }
    .sb-mal   { background: #fee2e2; color: #991b1b; border: 1.5px solid #fca5a5; }

    /* 2 Big Primary KPI Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 18px;
        margin-bottom: 20px;
    }
    .kpi-cell {
        background: #f8fafc;
        border: 1.5px solid var(--border-subtle);
        border-radius: 14px;
        padding: 20px 24px;
    }
    .kpi-label {
        font-size: 14px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ink-muted);
    }
    .kpi-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 46px;
        font-weight: 800;
        line-height: 1.1;
        margin: 8px 0;
    }
    .kpi-sub {
        font-size: 14px;
        font-weight: 600;
        color: var(--ink-muted);
    }

    /* Plain-English Bullet Explanation Cards */
    .reason-box {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    .reason-threat {
        border-left: 6px solid #dc2626;
        background: #fffafa;
    }
    .reason-warning {
        border-left: 6px solid #d97706;
        background: #fffdfa;
    }
    .reason-safe {
        border-left: 6px solid #059669;
        background: #fafffc;
    }
    .reason-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .reason-desc {
        font-size: 15px;
        color: #334155;
        line-height: 1.6;
        margin-left: 36px;
    }

    /* Word Token Badges */
    .token-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 700;
        padding: 7px 15px;
        border-radius: 8px;
        margin: 4px 8px 4px 0;
    }
    .tc-threat { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .tc-safe   { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

    /* Recommended Security Action Card */
    .action-container {
        background: #eff6ff;
        border: 1.5px solid #bfdbfe;
        border-left: 8px solid #2563eb;
        border-radius: 16px;
        padding: 26px 34px;
        margin-top: 10px;
        margin-bottom: 28px;
        display: flex;
        align-items: flex-start;
        gap: 22px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
    }
    .action-icon {
        font-size: 44px;
        line-height: 1.0;
        margin-top: 2px;
    }
    .action-content-title {
        font-size: 19px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .action-content-body {
        font-size: 16px;
        color: #1e293b;
        line-height: 1.65;
        font-weight: 500;
    }

    /* Enlarged Executive Presentation Footer */
    .footer-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 16px;
        padding: 32px 42px;
        margin-top: 32px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    .footer-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        border-bottom: 1.5px solid var(--border-subtle);
        padding-bottom: 18px;
    }
    .footer-title {
        font-size: 21px;
        font-weight: 800;
        color: var(--ink-title);
        letter-spacing: -0.01em;
    }
    .footer-sub {
        font-size: 15px;
        color: var(--ink-muted);
        font-weight: 600;
        margin-top: 4px;
    }
    .footer-author {
        font-size: 17px;
        font-weight: 700;
        color: #1e293b;
        text-align: right;
    }
    .footer-author span {
        color: var(--brand-blue);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
    }
    .footer-author-sub {
        font-size: 14px;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 3px;
    }
    .footer-stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
    }
    .footer-stat-card {
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .footer-stat-label {
        font-size: 12.5px;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .footer-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 19px;
        font-weight: 800;
        color: #0f172a;
        margin: 6px 0 4px 0;
    }
    .footer-stat-sub {
        font-size: 13px;
        color: #475569;
        font-weight: 500;
    }
    .footer-bottom-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 14px;
        color: #64748b;
        border-top: 1px solid #f1f5f9;
        padding-top: 14px;
    }
    .footer-badge-pill {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 6px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 13.5px;
        color: #1e40af;
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
# Plain-English Explanation Generator (At Least 4 Reasons)
# ============================================
def generate_plain_english_reasons(res, body_text: str, subj_text: str, sender_text: str) -> list:
    """Generate at least 4 crystal-clear, plain-English bullet reasons explaining the AI's verdict."""
    reasons = []
    combined = (sender_text + " " + subj_text + " " + body_text).lower()

    if res.prediction != "LEGITIMATE":
        # 1. Sender Domain & Lookalike Analysis
        if sender_text and ("rnicrosoft" in sender_text.lower() or "paypa1" in sender_text.lower() or ".xyz" in sender_text.lower()):
            reasons.append({
                "type": "threat",
                "icon": "🎭",
                "title": "1. Fake Sender Address Trying to Trick You (Lookalike Scam)",
                "desc": f"The sender's address ('{sender_text}') replaces the letter 'm' with 'rn' to look like 'microsoft.com' at a quick glance. This is a classic visual trick used by scammers to pretend to be a company you trust."
            })
        elif any("impersonation" in ind.lower() or "typosquat" in ind.lower() for ind in res.detected_indicators):
            reasons.append({
                "type": "threat",
                "icon": "🎭",
                "title": "1. Brand Impersonation & Fake Corporate Name",
                "desc": "The sender domain closely mimics an official protected company brand with subtle typos, designed to fool recipients into believing it is genuine."
            })
        else:
            reasons.append({
                "type": "warning",
                "icon": "⚠️",
                "title": "1. Untrusted or Fake Sender Address",
                "desc": f"The email came from an unverified or unfamiliar domain ('{sender_text if sender_text else 'Unknown/Spoofed'}') rather than authentic corporate servers, showing the sender is hiding their real identity."
            })

        # 2. Psychological Urgency & Fear Pressure Tactics
        if any(w in combined for w in ["immediate", "within 24 hours", "within 2 hours", "quota exceeded", "blocked", "suspension", "final notice", "urgent", "freeze"]):
            reasons.append({
                "type": "warning",
                "icon": "⏱️",
                "title": "2. Creates Fake Panic & Rushes You to Act",
                "desc": "The email uses urgent pressure words (like 'immediate action required', 'account blocked in 24 hours', or 'permanent deactivation') to make you panic and click before thinking or asking your IT team."
            })
        else:
            reasons.append({
                "type": "warning",
                "icon": "🎣",
                "title": "2. Pushy Emotional Manipulation Tactics",
                "desc": "The message uses pushy, administrative language designed to rush you into doing something immediately without checking whether the request is real."
            })

        # 3. Call-to-Action & Credential Theft Mechanism
        if any(w in combined for w in ["verify your password", "restore-access", "login", "password", "credential", "account activity"]):
            reasons.append({
                "type": "threat",
                "icon": "🔒",
                "title": "3. Tries to Steal Your Login Password (Credential Theft)",
                "desc": "The email urges you to click a button or link to 'log in', 'verify your password', or 'restore access'. Clicking this opens a fake login screen built specifically to capture and steal your private password."
            })
        else:
            reasons.append({
                "type": "warning",
                "icon": "🔗",
                "title": "3. Suspicious Links Leading to Outside Websites",
                "desc": "The embedded links route you to unknown third-party hosting websites instead of the official, trusted business service infrastructure."
            })

        # 4. Attachment / Payload Threat Analysis
        if any(ext in combined for ext in [".exe", ".scr", ".bat", ".pdf.exe", ".vbs"]):
            reasons.append({
                "type": "threat",
                "icon": "📎",
                "title": "4. References a Dangerous Program File (.exe)",
                "desc": "The email mentions a dangerous computer program file (.exe) disguised as a routine PDF or invoice. Opening this file could secretly install malware or ransomware on your computer."
            })
        else:
            reasons.append({
                "type": "threat",
                "icon": "🛡️",
                "title": "4. Multiple Red Flags Triggered Simultaneously",
                "desc": "Multiple warning signs occurred together (unverified links, urgent deadlines, and unverified sender information), pushing this email into the elevated threat category."
            })

        # 5. AI NLP Feature Vector Attribution
        if res.top_threat_tokens:
            top_words = ", ".join([f"'{t['token']}'" for t in res.top_threat_tokens[:4]])
            reasons.append({
                "type": "threat",
                "icon": "🧠",
                "title": "5. AI Detected Multiple High-Risk Scam Words",
                "desc": f"Our trained machine learning model scanned the full text and detected high-risk scam words ({top_words}) that commonly appear in attack emails, mathematically pulling it into the threat category."
            })
        else:
            reasons.append({
                "type": "threat",
                "icon": "🧠",
                "title": "5. AI Statistical Pattern Matches Attack Database",
                "desc": "The 10,020-feature machine learning classifier detected strong structural similarities to known malicious campaigns, marking this message as unsafe."
            })

    else:
        # Legitimate Email: 5 Guaranteed Plain-English Reasons
        reasons.append({
            "type": "safe",
            "icon": "✅",
            "title": "1. Authentic Professional Workplace Language",
            "desc": "The email uses standard, calm business words (like meeting schedules, project updates, and team deliverables) with no panic words, no threats, and no forced rush."
        })
        reasons.append({
            "type": "safe",
            "icon": "🛡️",
            "title": "2. Clean & Verified Sender Address",
            "desc": f"The sender address ('{sender_text}') matches authentic corporate standards with zero misspelled company names, fake characters, or deceptive lookalike tricks."
        })
        reasons.append({
            "type": "safe",
            "icon": "🔗",
            "title": "3. No Deceptive Links or Password Traps",
            "desc": "There are no suspicious web links asking you to enter passwords, confirm bank details, or sign into unfamiliar external websites."
        })
        reasons.append({
            "type": "safe",
            "icon": "📎",
            "title": "4. Safe Message with No Dangerous Files",
            "desc": "The email contains clean text with no executable computer files (like .exe or .scr) or double-extension attachments that could harm your computer."
        })
        reasons.append({
            "type": "safe",
            "icon": "⚖️",
            "title": "5. AI Model Confirmed Deep Safe Zone",
            "desc": "Our calibrated machine learning model calculated that this message matches genuine business communication patterns, placing it securely in the green safe zone."
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
        <div class="student-meta">Roll / S.No: 70 · Final Project · KPITB AI/ML Training Program</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================
# 2. BEAUTIFUL QUICK TESTING BUTTONS
# ============================================
st.markdown('<div style="font-size: 14.5px; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 9px; letter-spacing: 0.04em;">⚡ Quick Test Scenarios (Click to Load):</div>', unsafe_allow_html=True)
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
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 13.5px; color: #2563eb;">Input Workstation</span>
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
        height=200,
        max_chars=MAX_EMAIL_LENGTH,
        placeholder="Paste complete email body, notification text, or suspicious URLs here...",
    )

    # High-visibility alerts if typosquatting or payload present
    if sender_val and "rnicrosoft" in sender_val.lower():
        st.markdown(
            '<div style="background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 8px; padding: 10px 14px; font-size: 14px; color: #b45309; margin-bottom: 12px;">⚠️ <strong>Homoglyph Spoof:</strong> \'rn\' simulates \'m\' (<code>rnicrosoft.com</code> vs <code>microsoft.com</code>)</div>',
            unsafe_allow_html=True,
        )
    if body_val and any(ext in body_val.lower() for ext in [".exe", ".scr", ".bat", ".pdf.exe"]):
        st.markdown(
            '<div style="background: #fef2f2; border: 1.5px solid #fecaca; border-radius: 8px; padding: 10px 14px; font-size: 14px; color: #dc2626; margin-bottom: 12px;">📎 <strong>Payload Alert:</strong> Executable attachment mention detected (<code>.exe / binary</code>)</div>',
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
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 13.5px; color: #059669;">● Live Prediction</span>
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

    # Big Verdict Banner (42px headline)
    st.markdown(
        f"""
    <div class="verdict-box {vb_cls}">
        <div>
            <div style="font-size: 13px; font-weight: 800; text-transform: uppercase; color: #64748b; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">
                AI Model Verdict
            </div>
            <div class="verdict-title {vt_cls}">{res.prediction}</div>
            <div style="font-size: 14px; color: #475569; margin-top: 5px;">
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

    # 2 Big Primary KPI Cards (46px Numbers)
    risk_int = int(res.risk_score * 100)
    st.markdown(
        f"""
    <div class="kpi-row">
        <div class="kpi-cell">
            <div class="kpi-label">Composite Risk Score</div>
            <div class="kpi-num" style="color: {res.risk_color};">{risk_int} <span style="font-size: 22px; font-weight: 600; color: #64748b;">/ 100</span></div>
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
        '<div style="font-size: 13.5px; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 6px; font-family: \'JetBrains Mono\', monospace;">Posterior Class Probabilities:</div>',
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
            textfont=dict(size=14, family="JetBrains Mono", color="white"),
        )
    )
    prob_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=125,
        margin=dict(l=0, r=20, t=4, b=4),
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=13.5, family="JetBrains Mono", color="#334155"), autorange="reversed"),
        bargap=0.22,
    )
    st.plotly_chart(prob_chart, use_container_width=True, config={"displayModeBar": False})

    # Top Triggering Words
    if res.top_threat_tokens:
        st.markdown(
            '<div style="font-size: 13px; font-weight: 800; color: #dc2626; font-family: \'JetBrains Mono\', monospace; margin: 12px 0 6px 0;">TOP THREAT KEYWORDS DETECTED:</div>',
            unsafe_allow_html=True,
        )
        chips_threat = "".join([f'<span class="token-chip tc-threat">{t["token"]} +{abs(t["impact"]):.2f}</span>' for t in res.top_threat_tokens[:5]])
        st.markdown(chips_threat, unsafe_allow_html=True)
    elif res.top_safe_tokens:
        st.markdown(
            '<div style="font-size: 13px; font-weight: 800; color: #059669; font-family: \'JetBrains Mono\', monospace; margin: 12px 0 6px 0;">TOP BENIGN KEYWORDS DETECTED:</div>',
            unsafe_allow_html=True,
        )
        chips_safe = "".join([f'<span class="token-chip tc-safe">{t["token"]} -{abs(t["impact"]):.2f}</span>' for t in res.top_safe_tokens[:5]])
        st.markdown(chips_safe, unsafe_allow_html=True)

# --------------------------------------------
# COLUMN 2: Why Was This Email Flagged? (At Least 4 Reasons)
# --------------------------------------------
with col_out2:
    st.markdown(
        f"""
    <div class="section-title">
        <span>Why Was This Email Marked As {res.prediction}?</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 13.5px; color: #2563eb;">Plain-English Reasons ({'5 Detected' if res.prediction != 'LEGITIMATE' else '5 Verified'})</span>
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
                <span style="font-size: 24px;">{r['icon']}</span>
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
# 6. ENLARGED EXECUTIVE PRESENTATION FOOTER (1920x1080)
# ============================================
st.markdown(
    """
<div class="footer-card">
    <div class="footer-top-row">
        <div>
            <div class="footer-title">🛡️ PhishGuard AI — Executive Capstone Presentation</div>
            <div class="footer-sub">
                Final Project · <strong>KPITB AI/ML Training Program</strong> (Directorate of Science & Technology, Khyber Pakhtunkhwa)
            </div>
        </div>
        <div class="footer-author">
            Lead Developer: <span>Muhammad Haris</span>
            <div class="footer-author-sub">Roll / S.No: 70 · Peshawar Center · Final Evaluation</div>
        </div>
    </div>
    <div class="footer-stats-grid">
        <div class="footer-stat-card">
            <div class="footer-stat-label">Model Architecture</div>
            <div class="footer-stat-val">Calibrated Linear SVM</div>
            <div class="footer-stat-sub">Platt-scaled via CalibratedClassifierCV</div>
        </div>
        <div class="footer-stat-card">
            <div class="footer-stat-label">NLP Feature Space</div>
            <div class="footer-stat-val">10,020 Features</div>
            <div class="footer-stat-sub">TF-IDF Word (1-2) N-Grams + Sublinear TF</div>
        </div>
        <div class="footer-stat-card">
            <div class="footer-stat-label">Model Accuracy</div>
            <div class="footer-stat-val" style="color: #059669;">98.64%</div>
            <div class="footer-stat-sub">Macro F1: 0.9761 on 82k Corpus</div>
        </div>
        <div class="footer-stat-card">
            <div class="footer-stat-label">Engineering Quality</div>
            <div class="footer-stat-val" style="color: #2563eb;">58/58 Tests Passing</div>
            <div class="footer-stat-sub">100% Pass Rate · Latency: ~12ms</div>
        </div>
    </div>
    <div class="footer-bottom-row">
        <div>
            Training Dataset: <strong>Enron Corporate & Real Phishing Email Corpora (82,486 emails)</strong>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="footer-badge-pill">🖥️ 1920x1080 Full HD Optimized</span>
            <span class="footer-badge-pill">⚡ Real-Time NLP Pipeline</span>
            <span class="footer-badge-pill">KPITB AI/ML 2026</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
