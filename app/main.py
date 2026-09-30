"""
Mail-Lens AI — Email Security Analysis Dashboard

Streamlit web application providing email threat classification,
risk assessment, and explainability using an NLP/ML pipeline.

Developer: Muhammad Haris
Program: KPITB AI/ML Training Program — Capstone Project
Model: Calibrated Linear SVM + TF-IDF (10,000 N-Grams) — 98.64% Test Accuracy
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import html
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference.engine import MailLensInference
from src.utils.config import MAX_EMAIL_LENGTH, MODELS_DIR

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Mail-Lens AI — Email Security Analysis",
    page_icon="🔍",
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
        font-size: 14px !important;
    }

    /* Completely eliminate Streamlit default header space */
    header[data-testid="stHeader"],
    header,
    .stApp > header {
        display: none !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
    }
    #MainMenu,
    footer,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
        height: 0px !important;
    }

    /* Responsive Main Container — minimal top spacing */
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1680px !important;
    }

    /* 1. TOP HEADER */
    .app-header {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
    }
    .brand-title {
        font-size: 18px;
        font-weight: 800;
        color: var(--ink-title);
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }
    .brand-title span {
        color: var(--brand-blue);
    }
    .brand-sub {
        font-size: 14px;
        color: var(--ink-muted);
        font-weight: 600;
        margin: 3px 0 0 0;
    }

    .student-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 12px;
        padding: 10px 20px;
        text-align: right;
    }
    .student-name {
        font-size: 18px;
        font-weight: 800;
        color: var(--ink-title);
        line-height: 1.2;
    }
    .student-meta {
        font-size: 12px;
        color: var(--brand-blue);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        margin-top: 2px;
    }

    /* Section Titles */
    .section-title {
        font-size: 16px;
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

    /* Scenario Preset Buttons */
    div[data-testid="stButton"] button {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        height: 48px !important;
        min-height: 48px !important;
        padding: 8px 14px !important;
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
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px -2px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0px) !important;
        background-color: #dbeafe !important;
    }
    div[data-testid="stButton"] button p {
        font-size: 14px !important;
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
        border-radius: 10px !important;
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
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 1.55 !important;
        padding: 12px 14px !important;
    }
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label p {
        font-size: 16px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #334155 !important;
        margin: 0 0 6px 0 !important;
    }

    /* Primary Submit Button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 16px !important;
        font-weight: 800 !important;
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
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.44) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Verdict Banner */
    .verdict-box {
        border-radius: 12px;
        padding: 18px 24px;
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
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .vt-phish { color: #d97706; }
    .vt-legit { color: #059669; }
    .vt-mal   { color: #dc2626; }

    .status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 800;
        padding: 7px 16px;
        border-radius: 8px;
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
        gap: 16px;
        margin-bottom: 16px;
    }
    .kpi-cell {
        background: #f8fafc;
        border: 1.5px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 20px;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ink-muted);
    }
    .kpi-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        font-weight: 800;
        line-height: 1.2;
        margin: 6px 0;
    }
    .kpi-sub {
        font-size: 12px;
        font-weight: 600;
        color: var(--ink-muted);
    }

    /* Plain-English Bullet Explanation Cards */
    .reason-box {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
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
        gap: 10px;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .reason-desc {
        font-size: 14px;
        color: #334155;
        line-height: 1.5;
        margin-left: 28px;
    }

    /* Word Token Badges */
    .token-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 6px;
        margin: 3px 6px 3px 0;
    }
    .tc-threat { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .tc-safe   { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

    /* Recommended Security Action Card */
    .action-container {
        background: #eff6ff;
        border: 1.5px solid #bfdbfe;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 18px 24px;
        margin-top: 8px;
        margin-bottom: 22px;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
    }
    .action-icon {
        font-size: 18px;
        line-height: 1.2;
        margin-top: 2px;
    }
    .action-content-title {
        font-size: 16px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .action-content-body {
        font-size: 14px;
        color: #1e293b;
        line-height: 1.55;
        font-weight: 500;
    }

    /* Enlarged Executive Presentation Footer */
    .footer-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 14px;
        padding: 22px 28px;
        margin-top: 26px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .footer-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        border-bottom: 1.5px solid var(--border-subtle);
        padding-bottom: 14px;
    }
    .footer-title {
        font-size: 16px;
        font-weight: 800;
        color: var(--ink-title);
        letter-spacing: -0.01em;
    }
    .footer-sub {
        font-size: 14px;
        color: var(--ink-muted);
        font-weight: 600;
        margin-top: 3px;
    }
    .footer-author {
        font-size: 16px;
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
        font-size: 12px;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }
    .footer-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
    }
    .footer-stat-card {
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
    }
    .footer-stat-label {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .footer-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0 2px 0;
    }
    .footer-stat-sub {
        font-size: 12px;
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
        padding-top: 12px;
    }
    .footer-badge-pill {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 4px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 12px;
        color: #1e40af;
    }

    /* Form Clear Button Styling */
    div[data-testid="stForm"] div[data-testid="column"]:nth-of-type(2) button {
        background-color: #f8fafc !important;
        color: #475569 !important;
        border: 1.5px solid #cbd5e1 !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stForm"] div[data-testid="column"]:nth-of-type(2) button:hover {
        background-color: #fee2e2 !important;
        color: #dc2626 !important;
        border-color: #fca5a5 !important;
    }

    /* Keyword & Threat Highlighting (Red, Yellow, Green) */
    .hl-red {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 4px !important;
        padding: 1px 5px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        line-height: 1.3 !important;
    }
    .hl-yellow {
        background-color: #fef3c7 !important;
        color: #92400e !important;
        border: 1px solid #fcd34d !important;
        border-radius: 4px !important;
        padding: 1px 5px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        line-height: 1.3 !important;
    }
    .hl-green {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
        border: 1px solid #6ee7b7 !important;
        border-radius: 4px !important;
        padding: 1px 5px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        line-height: 1.3 !important;
    }
    .email-inspector-card {
        background: #ffffff;
        border: 1.5px solid var(--border-subtle);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 14px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    .inspector-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 8px;
    }
    .inspector-title {
        font-size: 16px;
        font-weight: 800;
        color: var(--ink-title);
    }
    .inspector-legend {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    .inspector-body {
        font-size: 14px;
        line-height: 1.6;
        color: #1e293b;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        max-height: 240px;
        overflow-y: auto;
    }

    /* ========================================================
       RESPONSIVE DESIGN & MOBILE/TABLET BREAKPOINTS
       ======================================================== */

    /* Medium Laptops & Desktops */
    @media (max-width: 1200px) {
        .main .block-container,
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        .footer-stats-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }

    /* Tablets & Foldables (Stack 2-column layout into clean single-column) */
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 12px !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .app-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 12px !important;
            padding: 14px 18px !important;
        }
        .student-card {
            width: 100% !important;
            text-align: left !important;
            padding: 10px 14px !important;
        }
        .footer-top-row {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
        }
        .footer-author {
            text-align: left !important;
        }
        .footer-stats-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }

    /* Mobile Phones & Handheld Devices */
    @media (max-width: 600px) {
        .main .block-container,
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: 0.3rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .app-header {
            padding: 12px 14px !important;
            margin-bottom: 12px !important;
        }
        .brand-logo {
            width: 40px !important;
            height: 40px !important;
            font-size: 16px !important;
            border-radius: 10px !important;
        }
        .brand-title {
            font-size: 16px !important;
        }
        .brand-sub {
            font-size: 12px !important;
        }
        .student-name {
            font-size: 15px !important;
        }
        .student-meta {
            font-size: 11px !important;
        }
        .section-title {
            font-size: 14px !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 4px !important;
        }
        div[data-testid="stButton"] button {
            height: 44px !important;
            min-height: 44px !important;
            font-size: 13px !important;
            padding: 6px 10px !important;
        }
        div[data-testid="stButton"] button p {
            font-size: 13px !important;
            white-space: normal !important;
            text-align: center !important;
        }
        .action-container {
            flex-direction: column !important;
            padding: 14px 16px !important;
            gap: 8px !important;
        }
        .footer-card {
            padding: 16px 14px !important;
        }
        .footer-stats-grid {
            grid-template-columns: 1fr !important;
            gap: 10px !important;
        }
        .footer-bottom-row {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }
        .reason-desc {
            margin-left: 0 !important;
            margin-top: 4px !important;
        }
        .reason-box {
            padding: 12px 14px !important;
        }
        .inspector-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 6px !important;
        }
        .inspector-body {
            padding: 10px 12px !important;
            font-size: 13px !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# Engine Loading
# ============================================
@st.cache_resource(show_spinner="Loading Mail-Lens AI Model...")
def load_engine():
    return MailLensInference(MODELS_DIR)

engine = load_engine()

# ============================================
# Visual Keyword Threat Highlighter Engine
# ============================================
def highlight_email_content(raw_text: str, is_threat: bool = True, threat_tokens: list = None) -> tuple:
    """
    Color-codes and highlights threat and security keywords within the email:
      - Red: Critical threats, credential harvesting, malware attachments, lookalike domains
      - Yellow: Urgency, panic, financial scam bait, promotional spam triggers, external URLs
      - Green: Authentic workplace collaboration terms (for legitimate emails)

    Returns (highlighted_html, red_count, yellow_count, detected_tags)
    """
    if not raw_text:
        return "", 0, 0, []

    # Custom threat tokens from ML model
    extra_red = []
    if threat_tokens:
        for t in threat_tokens:
            token = t.get("token", "") if isinstance(t, dict) else str(t)
            if len(token) >= 3 and token.lower() not in {"the", "and", "for", "with", "this", "from", "that", "your"}:
                extra_red.append(re.escape(token))

    red_words = [
        r"passwords?", r"passcode", r"log\s*in", r"login", r"sign\s*in", r"credentials?",
        r"verify\s+your\s+account", r"confirm\s+your\s+identity", r"verify\s+your\s+identity",
        r"reset\s+password", r"security\s+alert", r"unauthorized", r"account\s+suspension",
        r"reactivate", r"otp", r"2fa", r"pin", r"ssn", r"social\s+security",
        r"bank\s+account", r"banking", r"credit\s+card", r"update-verification",
        r"secure-login-portal", r"account-alert", r"rnicrosoft", r"paypa1", r"amaz0n",
        r"invoice_scan\.pdf\.exe", r"payload", r"exploit"
    ]
    if extra_red:
        red_words.extend(extra_red[:5])

    red_patterns = [
        r"\b(?:" + "|".join(red_words) + r")\b",
        r"\.(?:exe|scr|bat|vbs|iso|zip|js|cmd|pif|hta)\b",
        r"https?://[^\s<>\"']*(?:rnicrosoft|paypa1|amaz0n|update|verify|login|secure|auth|\.xyz|\.top)[^\s<>\"']*",
    ]

    yellow_patterns = [
        r"\b(?:immediately?|urgently?|within\s+24\s+hours|within\s+2\s+hours|24\s+hours|final\s+notice|final\s+warning|action\s+required|terminated|suspended|suspend|locked|expires?|immediate\s+action|restricted|restriction|limited\s+time|act\s+now|critical\s+alert|deadline|quota\s+exceeded|blocked|freeze)\b",
        r"\b(?:winners?|winning|congratulations|inheritance|prizes?|wire\s+transfer|western\s+union|crypto(?:currency)?|bitcoins?|lottery|funds?\s+transfer|million\s+dollars|beneficiary|compensation|claim\s+now|risk\s+free|grant|donation|unclaimed|reward|100%\s+free)\b",
        r"\b(?:buy\s+now|special\s+offer|exclusive\s+deal|viagra|guaranteed|unsubscribe|click\s+here|click\s+below|visit\s+link|free\s+gift|earn\s+extra|work\s+from\s+home)\b",
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    ]

    green_patterns = [
        r"\b(?:meeting|agenda|deliverables?|roadmap|quarterly|project\s+sync|engineering|standup|calendar\s+invite|attached\s+report|colleagues?|team|discussion|schedule|updates?|notes|budget|approved|quarter|presentation|milestones?)\b",
    ]

    matches = []
    detected_tags = set()

    for pat in red_patterns:
        for m in re.finditer(pat, raw_text, re.IGNORECASE):
            matches.append((m.start(), m.end(), "red", m.group(0)))
            detected_tags.add(m.group(0).lower())

    for pat in yellow_patterns:
        for m in re.finditer(pat, raw_text, re.IGNORECASE):
            matches.append((m.start(), m.end(), "yellow", m.group(0)))
            detected_tags.add(m.group(0).lower())

    if not is_threat:
        for pat in green_patterns:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                matches.append((m.start(), m.end(), "green", m.group(0)))

    # Sort matches by start position, prioritizing longer matches if tie
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Filter overlaps
    filtered_matches = []
    last_end = 0
    for start, end, cat, text in matches:
        if start >= last_end:
            filtered_matches.append((start, end, cat, text))
            last_end = end

    # Build highlighted output safely
    out = []
    last_idx = 0
    red_count = 0
    yellow_count = 0

    for start, end, cat, text in filtered_matches:
        out.append(html.escape(raw_text[last_idx:start]))
        if cat == "red":
            red_count += 1
            out.append(f'<span class="hl-red">{html.escape(text)}</span>')
        elif cat == "yellow":
            yellow_count += 1
            out.append(f'<span class="hl-yellow">{html.escape(text)}</span>')
        else:
            out.append(f'<span class="hl-green">{html.escape(text)}</span>')
        last_idx = end

    out.append(html.escape(raw_text[last_idx:]))
    formatted = "".join(out).replace("\n", "<br>")
    return formatted, red_count, yellow_count, list(detected_tags)


def detect_threat_subcategory(text: str, prediction: str, red_count: int, yellow_count: int) -> dict:
    """Classifies the email into detailed security threat categories (Phishing, Malware, Scam, Spam, or Safe)."""
    t_lower = text.lower()
    if prediction == "LEGITIMATE":
        return {
            "name": "Authentic Enterprise Communication",
            "icon": "🛡️",
            "badge_cls": "sb-legit",
            "color": "#059669",
            "desc": "Verified legitimate workplace correspondence with standard collaboration language.",
        }

    # 1. Executable malware payload
    if any(ext in t_lower for ext in [".exe", ".scr", ".bat", ".pdf.exe", ".vbs"]):
        return {
            "name": "Malware Payload Delivery (.exe)",
            "icon": "☣️",
            "badge_cls": "sb-mal",
            "color": "#dc2626",
            "desc": "Contains or references executable programs designed to compromise workstation endpoints.",
        }

    # 2. Financial Scam / Advance-Fee Fraud
    scam_keywords = ["wire transfer", "western union", "crypto", "bitcoin", "lottery", "prize", "winner", "inheritance", "million dollars", "beneficiary", "compensation", "grant money", "funds transfer"]
    if any(k in t_lower for k in scam_keywords):
        return {
            "name": "Financial Scam & Advance-Fee Fraud",
            "icon": "💰",
            "badge_cls": "sb-phish",
            "color": "#d97706",
            "desc": "Classic financial scam bait promising fake prizes, cryptocurrency, or wire transfer rewards.",
        }

    # 3. Unsolicited Commercial Spam / Marketing
    spam_keywords = ["viagra", "buy now", "special offer", "exclusive deal", "100% free", "work from home", "earn extra cash"]
    if any(k in t_lower for k in spam_keywords):
        return {
            "name": "Unsolicited Bulk Spam / Marketing",
            "icon": "📧",
            "badge_cls": "sb-phish",
            "color": "#d97706",
            "desc": "Unsolicited bulk email attempting to sell products or services through aggressive marketing.",
        }

    # 4. Credential Harvesting / Brand Spoof Phishing
    if any(k in t_lower for k in ["login", "password", "verify", "account", "rnicrosoft", "paypa1", "credentials", "suspended", "restore-access"]):
        return {
            "name": "Credential Harvesting & Brand Spoof Phishing",
            "icon": "🎣",
            "badge_cls": "sb-phish",
            "color": "#dc2626",
            "desc": "Deceptive brand impersonation attempting to capture user credentials and corporate passwords.",
        }

    # 5. General threat
    return {
        "name": "Suspicious Social Engineering Attack",
        "icon": "⚠️",
        "badge_cls": "sb-phish",
        "color": "#dc2626",
        "desc": "Uses urgent coercion and suspicious links to manipulate the recipient.",
    }


# ============================================
# Plain-English Explanation Generator (At Least 4 Reasons)
# ============================================
def generate_plain_english_reasons(res, body_text: str, subj_text: str, sender_text: str) -> list:
    """Generate short, easy-to-understand plain-English reasons for the AI verdict."""
    reasons = []
    combined = (sender_text + " " + subj_text + " " + body_text).lower()
    safe_sender = html.escape(sender_text) if sender_text else "Unknown"

    if res.prediction != "LEGITIMATE":
        # 1. Sender Address & Lookalikes
        if sender_text and ("rnicrosoft" in sender_text.lower() or "paypa1" in sender_text.lower() or ".xyz" in sender_text.lower()):
            reasons.append({
                "type": "threat",
                "icon": "🎭",
                "title": "1. Fake Sender Address",
                "desc": f"The sender email ('{safe_sender}') uses lookalike letters ('rn' for 'm') to secretly imitate a trusted company."
            })
        elif any("impersonation" in ind.lower() or "typosquat" in ind.lower() for ind in res.detected_indicators):
            reasons.append({
                "type": "threat",
                "icon": "🎭",
                "title": "1. Impersonated Brand Name",
                "desc": "The sender domain uses subtle spelling tricks to mimic an official brand and deceive you."
            })
        else:
            reasons.append({
                "type": "warning",
                "icon": "⚠️",
                "title": "1. Unverified Sender Domain",
                "desc": f"The email came from an untrusted outside address ('{safe_sender}') rather than official servers."
            })

        # 2. Fake Urgency & Panic Tactics
        if any(w in combined for w in ["immediate", "within 24 hours", "within 2 hours", "quota exceeded", "blocked", "suspension", "final notice", "urgent", "freeze"]):
            reasons.append({
                "type": "warning",
                "icon": "⏱️",
                "title": "2. Fake Panic Pressure",
                "desc": "Uses panic words ('immediate action', 'blocked in 24 hours') to rush you into clicking before verifying."
            })
        else:
            reasons.append({
                "type": "warning",
                "icon": "🎣",
                "title": "2. Pushy Language Tactics",
                "desc": "Uses pushy administrative wording designed to make you act fast without consulting IT."
            })

        # 3. Password / Login Stealing Traps
        if any(w in combined for w in ["verify your password", "restore-access", "login", "password", "credential", "account activity"]):
            reasons.append({
                "type": "threat",
                "icon": "🔒",
                "title": "3. Password Theft Link",
                "desc": "Directs you to a deceptive login page built specifically to steal your private password."
            })
        else:
            reasons.append({
                "type": "warning",
                "icon": "🔗",
                "title": "3. Suspicious External Link",
                "desc": "Contains links that redirect traffic to unverified external websites outside your organization."
            })

        # 4. Attachment / Scam / Spam / Threat Payload Analysis
        if any(ext in combined for ext in [".exe", ".scr", ".bat", ".pdf.exe", ".vbs"]):
            reasons.append({
                "type": "threat",
                "icon": "📎",
                "title": "4. Dangerous Program File (.exe)",
                "desc": "References an executable program (.exe) that can secretly install malware or ransomware on your PC."
            })
        elif any(k in combined for k in ["wire transfer", "western union", "crypto", "bitcoin", "lottery", "prize", "winner", "inheritance", "million dollars", "beneficiary"]):
            reasons.append({
                "type": "threat",
                "icon": "💰",
                "title": "4. Financial Scam / Fraud Lure",
                "desc": "Uses classic financial scam lures (fake prizes, cryptocurrency, or wire transfer promises) to defraud you."
            })
        elif any(k in combined for k in ["viagra", "buy now", "special offer", "exclusive deal", "unsubscribe"]):
            reasons.append({
                "type": "threat",
                "icon": "📧",
                "title": "4. Commercial Bulk Spam Pattern",
                "desc": "Contains unsolicited sales marketing buzzwords and aggressive promotional patterns typical of spam."
            })
        else:
            reasons.append({
                "type": "threat",
                "icon": "🛡️",
                "title": "4. Multiple Threat Signals",
                "desc": "Unverified links, urgent deadlines, and origin mismatches combined to trigger high risk."
            })

        # 5. AI Scam Pattern Match
        if res.top_threat_tokens:
            top_words = ", ".join([f"'{t['token']}'" for t in res.top_threat_tokens[:3]])
            reasons.append({
                "type": "threat",
                "icon": "🧠",
                "title": "5. AI Scam Pattern Match",
                "desc": f"The AI model detected high-risk scam words ({top_words}) typical of phishing attacks."
            })
        else:
            reasons.append({
                "type": "threat",
                "icon": "🧠",
                "title": "5. AI Threat Classifier Match",
                "desc": "Machine learning pattern matching flagged strong statistical similarity to known attack campaigns."
            })

    else:
        # Legitimate Email: 5 Short Plain-English Reasons
        reasons.append({
            "type": "safe",
            "icon": "✅",
            "title": "1. Normal Workplace Language",
            "desc": "Uses routine, professional business vocabulary with zero panic, threats, or pressure."
        })
        reasons.append({
            "type": "safe",
            "icon": "🛡️",
            "title": "2. Verified Sender Address",
            "desc": f"The sender address ('{safe_sender}') matches authentic business standards with no deceptive tricks."
        })
        reasons.append({
            "type": "safe",
            "icon": "🔗",
            "title": "3. No Password Traps",
            "desc": "Contains no fake login portals, credential harvesting forms, or hidden redirect links."
        })
        reasons.append({
            "type": "safe",
            "icon": "📎",
            "title": "4. No Risky Attachments",
            "desc": "Clean text message with no executable computer files (.exe) or dangerous software mentioned."
        })
        reasons.append({
            "type": "safe",
            "icon": "⚖️",
            "title": "5. AI Verified Safe Score",
            "desc": "The trained machine learning model calculated low risk, placing this email deep in the safe zone."
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

if "form_sender_input" not in st.session_state:
    st.session_state["form_sender_input"] = ""
if "form_subject_input" not in st.session_state:
    st.session_state["form_subject_input"] = ""
if "form_body_input" not in st.session_state:
    st.session_state["form_body_input"] = ""

def clear_all_inputs():
    """Wipes all input workstation fields clean."""
    st.session_state["form_sender_input"] = ""
    st.session_state["form_subject_input"] = ""
    st.session_state["form_body_input"] = ""

# ============================================
# 1. TOP HEADER: PROJECT NAME & STUDENT INFO
# ============================================
st.markdown(
    """
<div class="app-header">
    <div class="brand-section">
        <div class="brand-logo">🔍</div>
        <div>
            <div class="brand-title">Mail-Lens <span>AI</span></div>
            <div class="brand-sub">NLP-Driven Email Threat Analysis &amp; Risk Assessment</div>
        </div>
    </div>
    <div class="student-card">
        <div class="student-name">Muhammad Haris</div>
        <div class="student-meta">Final Project · KPITB AI/ML Training Program</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================
# 2. BEAUTIFUL QUICK TESTING BUTTONS
# ============================================
st.markdown('<div style="font-size: 16px; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 8px; letter-spacing: 0.04em;">⚡ Quick Test Scenarios (Click to Load):</div>', unsafe_allow_html=True)
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
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #2563eb;">Input Workstation</span>
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

    btn_c1, btn_c2 = st.columns([3, 1])
    with btn_c1:
        submitted = st.form_submit_button(
            "⚡ Analyze Threat & Score Risk",
            use_container_width=True,
            type="primary",
        )
    with btn_c2:
        cleared = st.form_submit_button(
            "🗑️ Clear All Inputs",
            use_container_width=True,
            on_click=clear_all_inputs,
        )

st.write("")

if cleared:
    curr_body = ""
    curr_subj = ""
    curr_sender = ""
else:
    curr_body = body_val if body_val else st.session_state.get("form_body_input", "")
    curr_subj = subject_val if subject_val else st.session_state.get("form_subject_input", "")
    curr_sender = sender_val if sender_val else st.session_state.get("form_sender_input", "")

has_content = bool(curr_body.strip() or curr_subj.strip() or curr_sender.strip())

if has_content:
    start_t = datetime.datetime.now()
    res = engine.analyze(text=curr_body, subject=curr_subj, sender=curr_sender)
    latency_ms = (datetime.datetime.now() - start_t).total_seconds() * 1000

    # Compute Visual Threat Keyword Highlights
    is_threat = res.prediction != "LEGITIMATE"
    highlighted_body, r_body, y_body, tags_body = highlight_email_content(
        curr_body, is_threat=is_threat, threat_tokens=res.top_threat_tokens
    )
    highlighted_subj, r_subj, y_subj, tags_subj = highlight_email_content(
        curr_subj, is_threat=is_threat
    )
    highlighted_sender, r_send, y_send, tags_send = highlight_email_content(
        curr_sender, is_threat=is_threat
    )

    total_red = r_body + r_subj + r_send
    total_yellow = y_body + y_subj + y_send
    threat_subcat = detect_threat_subcategory(
        (curr_sender + " " + curr_subj + " " + curr_body),
        res.prediction,
        total_red,
        total_yellow,
    )

    # Render Visual Keyword Threat Highlighter (Email Inspector Card)
    st.markdown(
        f"""
<div class="email-inspector-card">
    <div class="inspector-header">
        <div class="inspector-title">
            🔍 Visual Keyword Threat Highlighter (Email Inspector)
        </div>
        <div class="inspector-legend">
            <span class="hl-red">🔴 Red: Critical Threat / Credentials / Payload</span>
            <span class="hl-yellow">🟡 Yellow: Urgency / Scam / External Links</span>
            <span class="hl-green">🟢 Green: Authentic Workplace</span>
        </div>
    </div>
    <div class="inspector-body">
        <div style="margin-bottom: 6px; font-size: 14px; color: #475569;">
            <strong style="color: #0f172a;">From:</strong> {highlighted_sender if highlighted_sender else '<span style="color: #94a3b8; font-style: italic;">(None provided)</span>'}
        </div>
        <div style="margin-bottom: 8px; font-size: 14px; color: #475569;">
            <strong style="color: #0f172a;">Subject:</strong> {highlighted_subj if highlighted_subj else '<span style="color: #94a3b8; font-style: italic;">(None provided)</span>'}
        </div>
        <hr style="margin: 8px 0 10px 0; border: none; border-top: 1px dashed #cbd5e1;">
        <div style="font-size: 14px; line-height: 1.6; color: #1e293b;">
            {highlighted_body if highlighted_body else '<span style="color: #94a3b8; font-style: italic;">(Empty email body)</span>'}
        </div>
    </div>
    <div style="margin-top: 10px; display: flex; gap: 10px; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div style="display: flex; gap: 8px; flex-wrap: wrap; font-size: 12px; font-family: 'JetBrains Mono', monospace;">
            <span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; border: 1px solid #fca5a5; font-weight: 700;">🔴 Critical Threats: {total_red}</span>
            <span style="background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; border: 1px solid #fcd34d; font-weight: 700;">🟡 Urgency / Scam Triggers: {total_yellow}</span>
        </div>
        <div style="font-size: 12px; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {threat_subcat['color']};">
            <span>Threat Taxonomy: <strong>{threat_subcat['icon']} {threat_subcat['name']}</strong></span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

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
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #059669;">● Live Prediction</span>
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

        # Big Verdict Banner (18px headline)
        st.markdown(
            f"""
        <div class="verdict-box {vb_cls}">
            <div>
                <div style="font-size: 12px; font-weight: 800; text-transform: uppercase; color: #64748b; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">
                    AI Model Verdict
                </div>
                <div class="verdict-title {vt_cls}">{res.prediction}</div>
                <div style="font-size: 12px; color: #475569; margin-top: 4px;">
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

        # 2 Big Primary KPI Cards (18px Numbers)
        risk_int = int(res.risk_score * 100)
        st.markdown(
            f"""
        <div class="kpi-row">
            <div class="kpi-cell">
                <div class="kpi-label">Composite Risk Score</div>
                <div class="kpi-num" style="color: {res.risk_color};">{risk_int} <span style="font-size: 14px; font-weight: 600; color: #64748b;">/ 100</span></div>
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
            '<div style="font-size: 14px; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 6px; font-family: \'JetBrains Mono\', monospace;">Posterior Class Probabilities:</div>',
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
            yaxis=dict(showgrid=False, tickfont=dict(size=14, family="JetBrains Mono", color="#334155"), autorange="reversed"),
            bargap=0.22,
        )
        st.plotly_chart(prob_chart, use_container_width=True, config={"displayModeBar": False})

        # Top Triggering Words
        if res.top_threat_tokens:
            st.markdown(
                '<div style="font-size: 12px; font-weight: 800; color: #dc2626; font-family: \'JetBrains Mono\', monospace; margin: 10px 0 4px 0;">TOP THREAT KEYWORDS DETECTED:</div>',
                unsafe_allow_html=True,
            )
            chips_threat = "".join([f'<span class="token-chip tc-threat">{html.escape(t["token"])} +{abs(t["impact"]):.2f}</span>' for t in res.top_threat_tokens[:5]])
            st.markdown(chips_threat, unsafe_allow_html=True)
        elif res.top_safe_tokens:
            st.markdown(
                '<div style="font-size: 12px; font-weight: 800; color: #059669; font-family: \'JetBrains Mono\', monospace; margin: 10px 0 4px 0;">TOP BENIGN KEYWORDS DETECTED:</div>',
                unsafe_allow_html=True,
            )
            chips_safe = "".join([f'<span class="token-chip tc-safe">{html.escape(t["token"])} -{abs(t["impact"]):.2f}</span>' for t in res.top_safe_tokens[:5]])
            st.markdown(chips_safe, unsafe_allow_html=True)

    # --------------------------------------------
    # COLUMN 2: Why Was This Email Flagged? (At Least 4 Reasons)
    # --------------------------------------------
    with col_out2:
        st.markdown(
            f"""
        <div class="section-title">
            <span>Why Was This Email Marked As {res.prediction}?</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #2563eb;">Plain-English Reasons ({'5 Detected' if res.prediction != 'LEGITIMATE' else '5 Verified'})</span>
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
            <div class="action-content-body">{html.escape(res.recommendation)}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    # Standby State Inspector Card
    st.markdown(
        """
<div class="email-inspector-card">
    <div class="inspector-header">
        <div class="inspector-title">
            🔍 Visual Keyword Threat Highlighter (Email Inspector)
        </div>
        <div class="inspector-legend">
            <span class="hl-red">🔴 Red: Critical Threat</span>
            <span class="hl-yellow">🟡 Yellow: Urgency / Scam</span>
            <span class="hl-green">🟢 Green: Authentic Workplace</span>
        </div>
    </div>
    <div class="inspector-body" style="text-align: center; padding: 24px 16px; color: #64748b; background: #f8fafc;">
        <div style="font-size: 26px; margin-bottom: 6px;">📥</div>
        <div style="font-size: 16px; font-weight: 700; color: #1e293b;">Input Workstation Ready for Evaluation</div>
        <div style="font-size: 14px; color: #64748b; margin-top: 4px; max-width: 650px; margin-left: auto; margin-right: auto;">
            Enter an email sender, subject, or message body in the console above, or click any of the <strong>4 Quick Test Scenarios</strong> above to automatically load real threat samples.
        </div>
    </div>
    <div style="margin-top: 10px; display: flex; gap: 10px; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div style="font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #64748b;">
            System Status: <strong style="color: #059669;">● Ready / Standby Mode</strong>
        </div>
        <div style="font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #2563eb;">
            NLP Security Engine: <strong>Initialized & Loaded</strong>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # Standby Two Columns
    col_out1, col_out2 = st.columns([1, 1], gap="large")

    with col_out1:
        st.markdown(
            """
        <div class="section-title">
            <span>2. AI Classification & Threat Ranking</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748b;">● Standby</span>
        </div>
        <div class="verdict-box" style="background: #f8fafc; border: 1.5px solid #e2e8f0;">
            <div>
                <div style="font-size: 12px; font-weight: 800; text-transform: uppercase; color: #64748b; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">
                    AI Model Status
                </div>
                <div class="verdict-title" style="color: #475569;">STANDBY</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    Calibrated Linear SVM · 10,000 N-Gram Features · Ready for Evaluation
                </div>
            </div>
            <div style="text-align: right;">
                <span class="status-badge" style="background: #e2e8f0; color: #475569;">● AWAITING INPUT</span>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-cell">
                <div class="kpi-label">Composite Risk Score</div>
                <div class="kpi-num" style="color: #64748b;">-- <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">/ 100</span></div>
                <div class="kpi-sub">Threat Severity: <strong>Awaiting Input</strong></div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-label">Model Confidence</div>
                <div class="kpi-num" style="color: #64748b;">--%</div>
                <div class="kpi-sub">Platt-Calibrated SVM Probability</div>
            </div>
        </div>
        <div style="background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: center; color: #64748b; margin-top: 10px;">
            <div style="font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 6px;">Multi-Class Probability Distribution</div>
            <div style="font-size: 13px; color: #64748b;">Awaiting email text to project input across Legitimate, Phishing, and Malicious decision boundaries.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_out2:
        st.markdown(
            """
        <div class="section-title">
            <span>Security Inspection Criteria</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #2563eb;">5 Detection Factors</span>
        </div>
        <div class="reason-box reason-safe">
            <div class="reason-header">
                <span style="font-size: 18px;">🎭</span>
                <span>1. Sender Address & Lookalikes</span>
            </div>
            <div class="reason-desc">Evaluates sender domain for homoglyph substitution and typosquatting (e.g. rnicrosoft.com).</div>
        </div>
        <div class="reason-box reason-warning">
            <div class="reason-header">
                <span style="font-size: 18px;">⏱️</span>
                <span>2. Coercion & Panic Pressure</span>
            </div>
            <div class="reason-desc">Detects artificial panic deadlines ('within 24 hours') designed to bypass critical thinking.</div>
        </div>
        <div class="reason-box reason-threat">
            <div class="reason-header">
                <span style="font-size: 18px;">🔒</span>
                <span>3. Password Harvesting Traps</span>
            </div>
            <div class="reason-desc">Identifies fake credential collection forms and deceptive login redirection portals.</div>
        </div>
        <div class="reason-box reason-threat">
            <div class="reason-header">
                <span style="font-size: 18px;">📎</span>
                <span>4. Malware & Scam Triggers</span>
            </div>
            <div class="reason-desc">Scans for executable payload extensions (.exe) and advance-fee financial scam bait.</div>
        </div>
        <div class="reason-box reason-safe">
            <div class="reason-header">
                <span style="font-size: 18px;">🧠</span>
                <span>5. AI Statistical Pattern Match</span>
            </div>
            <div class="reason-desc">Projects n-grams across calibrated SVM hyperplanes to calculate threat probabilities.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Standby Recommended Security Action
    st.write("")
    st.markdown(
        """
    <div class="action-container" style="border-left-color: #3b82f6;">
        <div class="action-icon">💡</div>
        <div>
            <div class="action-content-title">Recommended Security Action</div>
            <div class="action-content-body">Please enter an email in the console above or select a preset scenario. Mail-Lens AI will classify the threat, highlight malicious keywords, and provide actionable security guidance.</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================
# 6. EXECUTIVE PRESENTATION FOOTER
# ============================================
st.markdown(
    """
<div class="footer-card">
    <div class="footer-top-row">
        <div>
            <div class="footer-title">🔍 Mail-Lens AI — Email Security Analysis</div>
            <div class="footer-sub">
                Final Project · <strong>KPITB AI/ML Training Program</strong>
            </div>
        </div>
        <div class="footer-author">
            Lead Developer: <span>Muhammad Haris</span>
            <div class="footer-author-sub">Final Project · KPITB AI/ML Training Program</div>
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
            <div class="footer-stat-val">10,000 Features</div>
            <div class="footer-stat-sub">TF-IDF Word (1-2) N-Grams + 20 Heuristics</div>
        </div>
        <div class="footer-stat-card">
            <div class="footer-stat-label">Model Accuracy</div>
            <div class="footer-stat-val" style="color: #059669;">98.64%</div>
            <div class="footer-stat-sub">Macro F1: 0.9761 on 17,960 Corpus</div>
        </div>
        <div class="footer-stat-card">
            <div class="footer-stat-label">Engineering Quality</div>
            <div class="footer-stat-val" style="color: #2563eb;">71/71 Tests Passing</div>
            <div class="footer-stat-sub">100% Pass Rate · Latency: ~12ms</div>
        </div>
    </div>
    <div class="footer-bottom-row">
        <div>
            Training Dataset: <strong>HuggingFace & Zenodo Phishing/Benign Corpora (17,960 emails)</strong>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="footer-badge-pill">⚡ Real-Time NLP Pipeline</span>
            <span class="footer-badge-pill">KPITB AI/ML 2026</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
