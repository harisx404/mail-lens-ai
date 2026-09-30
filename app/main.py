"""
PhishGuard AI — Streamlit Web Application (v2.0)

Professional SOC (Security Operations Center) Dashboard.
Dark cybersecurity theme with Plotly charts, animated risk gauge,
and production-grade UI.

Usage:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import json
import datetime
import plotly.graph_objects as go
from src.inference.engine import PhishGuardInference, AnalysisResult
from src.utils.config import MODELS_DIR, MAX_EMAIL_LENGTH

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="PhishGuard AI — Email Threat Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Professional Dark SOC Theme
# ============================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* ===== Base Theme ===== */
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: rgba(30, 41, 59, 0.8);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-cyan: #06b6d4;
        --accent-blue: #3b82f6;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --accent-purple: #8b5cf6;
        --border: #334155;
    }

    /* Global overrides */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1300px;
    }

    /* ===== Header Banner ===== */
    .soc-header {
        background: linear-gradient(135deg, #0c1220 0%, #162033 30%, #1a2845 60%, #0f172a 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(6, 182, 212, 0.15);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        position: relative;
        overflow: hidden;
    }
    .soc-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .soc-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f1f5f9;
    }
    .soc-header .subtitle {
        margin: 4px 0 0 0;
        font-size: 14px;
        color: #64748b;
        font-weight: 400;
    }
    .soc-header .brand-accent {
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ===== Metric Cards ===== */
    .metric-card-v2 {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 18px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }
    .metric-card-v2:hover {
        border-color: rgba(6, 182, 212, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    .metric-value-v2 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 36px;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-label-v2 {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ===== Classification Result Card ===== */
    .result-card-v2 {
        border-radius: 14px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid;
        backdrop-filter: blur(8px);
        position: relative;
        overflow: hidden;
    }
    .result-card-v2::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .result-legitimate {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.25);
    }
    .result-legitimate::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .result-phishing {
        background: rgba(245, 158, 11, 0.08);
        border-color: rgba(245, 158, 11, 0.25);
    }
    .result-phishing::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .result-malicious {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.25);
    }
    .result-malicious::before { background: linear-gradient(90deg, #ef4444, #f87171); }

    .result-card-v2 .classification-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .result-card-v2 .classification-value {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .result-card-v2 .confidence-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        color: #94a3b8;
    }

    .legit-text { color: #10b981; }
    .phish-text { color: #f59e0b; }
    .mal-text { color: #ef4444; }

    /* ===== Risk Badge ===== */
    .risk-badge-v2 {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'JetBrains Mono', monospace;
    }
    .risk-low { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .risk-medium { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .risk-high { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); }
    .risk-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* ===== Indicator Cards ===== */
    .indicator-card {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 14px;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .indicator-card .indicator-icon {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        background: rgba(245, 158, 11, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
    }

    /* ===== Section Title ===== */
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 20px 0 10px 0;
        letter-spacing: -0.3px;
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, #334155 0%, transparent 100%);
        margin: 20px 0;
    }

    /* ===== System Info Cards ===== */
    .info-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
    }
    .info-card h4 {
        font-size: 13px;
        font-weight: 600;
        color: #06b6d4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .info-card ul { list-style: none; padding: 0; margin: 0; }
    .info-card ul li {
        padding: 4px 0;
        font-size: 14px;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .info-card ul li .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-green { background: #10b981; }
    .dot-amber { background: #f59e0b; }
    .dot-red { background: #ef4444; }
    .dot-cyan { background: #06b6d4; }

    /* ===== Recommendation Box ===== */
    .rec-box {
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid;
        margin: 8px 0;
    }
    .rec-safe {
        background: rgba(16, 185, 129, 0.06);
        border-color: rgba(16, 185, 129, 0.2);
        color: #a7f3d0;
    }
    .rec-warning {
        background: rgba(245, 158, 11, 0.06);
        border-color: rgba(245, 158, 11, 0.2);
        color: #fde68a;
    }
    .rec-danger {
        background: rgba(239, 68, 68, 0.06);
        border-color: rgba(239, 68, 68, 0.2);
        color: #fecaca;
    }

    /* ===== History Expanders ===== */
    .history-item {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        font-size: 14px;
    }

    /* ===== Sidebar Styling ===== */
    section[data-testid="stSidebar"] > div {
        background: #0c1220;
        border-right: 1px solid #1e293b;
    }

    /* ===== Hide Streamlit defaults ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #1e293b;
    }

    /* Fix Streamlit text input styling */
    .stTextInput input, .stTextArea textarea {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 1px #06b6d4 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Session State Initialization
# ============================================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = None
    st.session_state.model_loaded = False

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0, "legitimate": 0, "phishing": 0,
        "malicious": 0, "high_risk": 0,
    }


def load_inference_engine():
    """Load the inference engine if not already loaded."""
    if not st.session_state.model_loaded:
        try:
            engine = PhishGuardInference()
            engine.load_model(MODELS_DIR)
            st.session_state.inference_engine = engine
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Model not available: {e}")
            st.info(
                "Please train the model first by running:\n"
                "```\npython scripts/train.py\n```"
            )
            return False
    return True


def render_header():
    """Render the professional SOC header."""
    st.markdown("""
    <div class="soc-header">
        <h1><span class="brand-accent">PhishGuard</span> AI</h1>
        <p class="subtitle">NLP-Based Email Threat Detection & Risk Analysis System</p>
    </div>
    """, unsafe_allow_html=True)


def create_risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    """Create an animated risk gauge using Plotly."""
    color_map = {
        "LOW": "#10b981",
        "MEDIUM": "#f59e0b",
        "HIGH": "#f97316",
        "CRITICAL": "#ef4444",
    }
    gauge_color = color_map.get(risk_level, "#64748b")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9", "family": "JetBrains Mono"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "dtick": 25,
                "tickfont": {"size": 10, "color": "#64748b"},
            },
            "bar": {"color": gauge_color, "thickness": 0.85},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.08)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.08)"},
                {"range": [60, 85], "color": "rgba(249, 115, 22, 0.08)"},
                {"range": [85, 100], "color": "rgba(239, 68, 68, 0.08)"},
            ],
            "threshold": {
                "line": {"color": gauge_color, "width": 3},
                "thickness": 0.85,
                "value": risk_score * 100,
            },
        },
        title={"text": f"<b>{risk_level} RISK</b>", "font": {"size": 14, "color": gauge_color}},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create a horizontal bar chart for class probabilities."""
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    colors = {
        "LEGITIMATE": "#10b981",
        "PHISHING": "#f59e0b",
        "MALICIOUS": "#ef4444",
    }
    bar_colors = [colors.get(c, "#64748b") for c in classes]

    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="inside",
        textfont=dict(size=13, family="JetBrains Mono", color="white"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#94a3b8", family="Inter"),
            autorange="reversed",
        ),
        bargap=0.35,
    )
    return fig


# ============================================
# Sidebar Navigation
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <span style="font-size: 28px; font-weight: 800; letter-spacing: -1px;">
            <span style="background: linear-gradient(135deg, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">PhishGuard</span>
            <span style="color: #f1f5f9;"> AI</span>
        </span>
        <p style="font-size: 11px; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 2px;">Threat Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Email", "History", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status
    if st.session_state.model_loaded:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 10px 14px;">
            <div style="font-size: 12px; color: #10b981; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">System Online</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Model loaded & ready</div>
        </div>
        """, unsafe_allow_html=True)
        try:
            metadata_path = MODELS_DIR / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    meta = json.load(f)
                st.caption(f"Model: {meta.get('model_name', 'Unknown')}")
                acc = meta.get('test_accuracy', 'N/A')
                if isinstance(acc, float):
                    st.caption(f"Accuracy: {acc:.2%}")
        except Exception:
            pass
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 8px; padding: 10px 14px;">
            <div style="font-size: 12px; color: #f59e0b; font-weight: 600;">Model Not Loaded</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 11px; color: #475569;">
        <strong style="color: #64748b;">PhishGuard AI</strong> v2.0<br>
        Muhammad Haris · KPITB
    </div>
    """, unsafe_allow_html=True)


# ============================================
# PAGE: Dashboard
# ============================================
if page == "Dashboard":
    render_header()

    # Stats cards
    stats = st.session_state.stats
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card-v2">
            <div class="metric-value-v2 cyan">{stats["total"]}</div>
            <div class="metric-label-v2">Total Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card-v2">
            <div class="metric-value-v2 emerald">{stats["legitimate"]}</div>
            <div class="metric-label-v2">Legitimate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card-v2">
            <div class="metric-value-v2 amber">{stats["phishing"]}</div>
            <div class="metric-label-v2">Phishing</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card-v2">
            <div class="metric-value-v2 red">{stats["malicious"]}</div>
            <div class="metric-label-v2">Malicious</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card-v2">
            <div class="metric-value-v2 purple">{stats["high_risk"]}</div>
            <div class="metric-label-v2">High Risk</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Recent analyses
    st.markdown('<div class="section-title">Recent Analyses</div>', unsafe_allow_html=True)

    if st.session_state.analysis_history:
        for item in reversed(st.session_state.analysis_history[-10:]):
            result = item["result"]
            risk_class = f"risk-{result.risk_level.lower()}"

            col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
            with col_a:
                st.text(result.email_snippet[:60])
            with col_b:
                pred_cls = result.prediction.lower()
                color_cls = {"legitimate": "legit", "phishing": "phish", "malicious": "mal"}.get(pred_cls, "")
                st.markdown(f'<strong class="{color_cls}-text">{result.prediction}</strong>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f"`{result.confidence:.1%}`")
            with col_d:
                st.markdown(
                    f'<span class="risk-badge-v2 {risk_class}">{result.risk_level}</span>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No analyses yet. Go to **Analyze Email** to start.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # System info
    st.markdown('<div class="section-title">System Information</div>', unsafe_allow_html=True)
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        <div class="info-card">
            <h4>Classification Categories</h4>
            <ul>
                <li><span class="dot dot-green"></span> <strong style="color: #f1f5f9;">LEGITIMATE</strong> — Normal, non-malicious email</li>
                <li><span class="dot dot-amber"></span> <strong style="color: #f1f5f9;">PHISHING</strong> — Social engineering / deception attempt</li>
                <li><span class="dot dot-red"></span> <strong style="color: #f1f5f9;">MALICIOUS</strong> — Harmful payload / malware delivery</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_info2:
        st.markdown("""
        <div class="info-card">
            <h4>Risk Levels</h4>
            <ul>
                <li><span class="dot dot-green"></span> <strong style="color: #f1f5f9;">LOW</strong> — Minimal concern</li>
                <li><span class="dot dot-amber"></span> <strong style="color: #f1f5f9;">MEDIUM</strong> — Review recommended</li>
                <li><span class="dot" style="background: #f97316;"></span> <strong style="color: #f1f5f9;">HIGH</strong> — Significant risk indicators</li>
                <li><span class="dot dot-red"></span> <strong style="color: #f1f5f9;">CRITICAL</strong> — Strong threat indicators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# PAGE: Analyze Email
# ============================================
elif page == "Analyze Email":
    render_header()
    st.markdown('<div class="section-title">Email Analysis</div>', unsafe_allow_html=True)

    if not load_inference_engine():
        st.stop()

    # Input form
    with st.form("email_form", clear_on_submit=False):
        col_subj, col_sender = st.columns(2)
        with col_subj:
            subject = st.text_input(
                "Email Subject",
                placeholder="e.g., Urgent: Verify Your Account Now",
            )
        with col_sender:
            sender = st.text_input(
                "Sender (optional)",
                placeholder="e.g., support@example.com",
            )

        body = st.text_area(
            "Email Body",
            height=180,
            max_chars=MAX_EMAIL_LENGTH,
            placeholder="Paste the email content here for threat analysis...",
        )

        submitted = st.form_submit_button(
            "Analyze Email",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not body and not subject:
            st.error("Please provide an email subject or body for analysis.")
        else:
            with st.spinner("Analyzing email for threats..."):
                result = st.session_state.inference_engine.analyze(
                    text=body, subject=subject, sender=sender
                )

            # Store in history
            st.session_state.analysis_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "result": result,
            })

            # Update stats
            st.session_state.stats["total"] += 1
            pred_lower = result.prediction.lower()
            if pred_lower in st.session_state.stats:
                st.session_state.stats[pred_lower] += 1
            if result.risk_level in ("HIGH", "CRITICAL"):
                st.session_state.stats["high_risk"] += 1

            # Display results
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

            # Prediction card
            pred_class_map = {
                "LEGITIMATE": ("result-legitimate", "legit-text"),
                "PHISHING": ("result-phishing", "phish-text"),
                "MALICIOUS": ("result-malicious", "mal-text"),
            }
            card_cls, text_cls = pred_class_map.get(result.prediction, ("", ""))

            st.markdown(f"""
            <div class="result-card-v2 {card_cls}">
                <div class="classification-label">Classification</div>
                <div class="classification-value {text_cls}">{result.prediction}</div>
                <div class="confidence-value">Confidence: {result.confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            # Risk gauge + Probabilities
            col_risk, col_prob = st.columns(2)

            with col_risk:
                st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
                risk_gauge = create_risk_gauge(result.risk_score, result.risk_level)
                st.plotly_chart(risk_gauge, use_container_width=True, config={"displayModeBar": False})
                st.caption(
                    "Risk assessment is application-level, not an objectively validated security rating."
                )

            with col_prob:
                st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)
                prob_chart = create_probability_chart(result.probabilities)
                st.plotly_chart(prob_chart, use_container_width=True, config={"displayModeBar": False})

            # Security Indicators
            if result.detected_indicators:
                st.markdown('<div class="section-title">Detected Security Indicators</div>', unsafe_allow_html=True)
                for indicator in result.detected_indicators:
                    icon = "⚠️"
                    if "brand impersonation" in indicator.lower():
                        icon = "🔍"
                    elif "simulation" in indicator.lower() or "cloaking" in indicator.lower():
                        icon = "🎭"
                    elif "url" in indicator.lower():
                        icon = "🔗"
                    elif "executable" in indicator.lower():
                        icon = "📎"

                    st.markdown(f"""
                    <div class="indicator-card">
                        <div class="indicator-icon">{icon}</div>
                        <span>{indicator}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Explanation
            st.markdown('<div class="section-title">Explanation</div>', unsafe_allow_html=True)
            st.info(result.explanation)

            # Recommendation
            st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)
            if result.prediction == "LEGITIMATE" and result.risk_level in ("LOW", "MEDIUM"):
                rec_class = "rec-safe"
            elif result.prediction == "LEGITIMATE":
                rec_class = "rec-warning"
            elif result.prediction == "PHISHING":
                rec_class = "rec-warning"
            else:
                rec_class = "rec-danger"

            st.markdown(f"""
            <div class="rec-box {rec_class}">{result.recommendation}</div>
            """, unsafe_allow_html=True)


# ============================================
# PAGE: History
# ============================================
elif page == "History":
    render_header()
    st.markdown('<div class="section-title">Analysis History</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_history:
        st.info("No analysis history yet.")
    else:
        # Clear button
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.session_state.stats = {
                "total": 0, "legitimate": 0, "phishing": 0,
                "malicious": 0, "high_risk": 0,
            }
            st.rerun()

        # History table
        for i, item in enumerate(reversed(st.session_state.analysis_history)):
            result = item["result"]
            timestamp = item["timestamp"][:19].replace("T", " ")

            pred_cls = result.prediction.lower()
            color_map = {"legitimate": "#10b981", "phishing": "#f59e0b", "malicious": "#ef4444"}
            pred_color = color_map.get(pred_cls, "#64748b")

            with st.expander(
                f"{timestamp}  |  {result.prediction}  |  "
                f"{result.confidence:.1%}  |  {result.risk_level}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Prediction:** <span style='color: {pred_color}'>{result.prediction}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** `{result.confidence:.1%}`")
                with col2:
                    risk_class = f"risk-{result.risk_level.lower()}"
                    st.markdown(f'**Risk:** <span class="risk-badge-v2 {risk_class}">{result.risk_level}</span>', unsafe_allow_html=True)
                    st.markdown(f"**Score:** `{result.risk_score:.3f}`")
                with col3:
                    st.markdown(f"**Indicators:** `{result.indicator_count}`")

                if result.detected_indicators:
                    st.markdown("**Detected Indicators:**")
                    for ind in result.detected_indicators:
                        st.markdown(f"- {ind}")

                st.markdown(f"**Snippet:** {result.email_snippet}")


# ============================================
# PAGE: About
# ============================================
elif page == "About":
    render_header()

    st.markdown("""
    <div class="section-title">About PhishGuard AI</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **PhishGuard AI** is an NLP-based email security analysis system developed as an
    AI/ML capstone project. It combines Natural Language Processing with cybersecurity
    domain knowledge to classify emails and assess security risks.
    """)

    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)

    # Architecture pipeline visualization
    fig = go.Figure()
    steps = ["Email Input", "NLP\nPreprocessing", "Feature\nExtraction", "ML\nClassification", "Risk\nAssessment", "Dashboard\nOutput"]
    colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

    for i, (step, color) in enumerate(zip(steps, colors)):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode="markers+text",
            marker=dict(size=50, color=color, opacity=0.15),
            text=[step],
            textposition="middle center",
            textfont=dict(size=11, color=color, family="Inter"),
            showlegend=False,
        ))
        if i < len(steps) - 1:
            fig.add_annotation(
                x=i + 0.5, y=0,
                text="→", font=dict(size=20, color="#334155"),
                showarrow=False,
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=100, margin=dict(l=20, r=20, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div class="info-card">
            <h4>Classification Categories</h4>
            <ul>
                <li><span class="dot dot-green"></span> <strong style="color: #f1f5f9;">LEGITIMATE</strong> — Normal, non-malicious email</li>
                <li><span class="dot dot-amber"></span> <strong style="color: #f1f5f9;">PHISHING</strong> — Social engineering / deception attempt</li>
                <li><span class="dot dot-red"></span> <strong style="color: #f1f5f9;">MALICIOUS</strong> — Harmful payload / malware delivery</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown("""
        <div class="info-card">
            <h4>Technologies</h4>
            <ul>
                <li><span class="dot dot-cyan"></span> <strong style="color: #f1f5f9;">NLP:</strong> TF-IDF, NLTK, lemmatization</li>
                <li><span class="dot dot-cyan"></span> <strong style="color: #f1f5f9;">ML:</strong> Logistic Regression, Naive Bayes, Linear SVM</li>
                <li><span class="dot dot-cyan"></span> <strong style="color: #f1f5f9;">Security:</strong> URL analysis, keyword indicators, typosquatting</li>
                <li><span class="dot dot-cyan"></span> <strong style="color: #f1f5f9;">Web:</strong> Streamlit, Plotly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Important Disclaimers</h4>
        <ul>
            <li><span class="dot dot-amber"></span> This is a <strong style="color: #f1f5f9;">research/educational capstone prototype</strong></li>
            <li><span class="dot dot-amber"></span> NOT a replacement for professional email security products</li>
            <li><span class="dot dot-amber"></span> All analysis is performed <strong style="color: #f1f5f9;">locally</strong> — no URLs visited, no files executed</li>
            <li><span class="dot dot-amber"></span> False positives and false negatives are possible and documented</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 16px 0;">
        <span style="color: #64748b; font-size: 13px;">
            <strong style="color: #94a3b8;">PhishGuard AI</strong> v2.0 · Developed by <strong style="color: #06b6d4;">Muhammad Haris</strong><br>
            AI/ML Training Program — KPITB
        </span>
    </div>
    """, unsafe_allow_html=True)
