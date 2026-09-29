"""
PhishGuard AI — Streamlit Web Application

Main entry point for the web interface.
Provides: Dashboard, Email Analysis, History pages.

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
from src.inference.engine import PhishGuardInference, AnalysisResult
from src.utils.config import MODELS_DIR, MAX_EMAIL_LENGTH

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Custom CSS for Professional Styling
# ============================================
st.markdown("""
<style>
    /* Main theme */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .phishguard-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .phishguard-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .phishguard-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }

    /* Result card */
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .result-legitimate {
        background: #0d3321;
        border-color: #28a745;
        color: #d4edda;
    }
    .result-phishing {
        background: #3d1f00;
        border-color: #fd7e14;
        color: #ffe0b2;
    }
    .result-malicious {
        background: #3d0000;
        border-color: #dc3545;
        color: #f8d7da;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-low { background: #28a745; color: white; }
    .risk-medium { background: #ffc107; color: black; }
    .risk-high { background: #fd7e14; color: white; }
    .risk-critical { background: #dc3545; color: white; }

    /* Indicator list */
    .indicator-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-left: 3px solid #ffc107;
        font-size: 0.9rem;
    }

    /* History table */
    .history-row {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
            st.error(f"⚠️ Model not available: {e}")
            st.info(
                "Please train the model first by running:\n"
                "```\npython scripts/train.py\n```"
            )
            return False
    return True


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="phishguard-header">
        <h1>🛡️ PhishGuard AI</h1>
        <p>NLP-Based Phishing & Malicious Email Detection and Risk Analysis System</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# Sidebar Navigation
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("PhishGuard AI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Analyze Email", "📋 History", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded")
        try:
            metadata_path = MODELS_DIR / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    meta = json.load(f)
                st.caption(f"Model: {meta.get('model_name', 'Unknown')}")
                st.caption(f"Accuracy: {meta.get('test_accuracy', 'N/A')}")
        except Exception:
            pass
    else:
        st.warning("⚠️ Model not loaded")

    st.markdown("---")
    st.caption("PhishGuard AI v1.0")
    st.caption("Capstone Project — Muhammad Haris")


# ============================================
# PAGE: Dashboard
# ============================================
if page == "🏠 Dashboard":
    render_header()

    # Stats cards
    stats = st.session_state.stats
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Analyzed", stats["total"])
    with col2:
        st.metric("✅ Legitimate", stats["legitimate"])
    with col3:
        st.metric("🎣 Phishing", stats["phishing"])
    with col4:
        st.metric("💀 Malicious", stats["malicious"])
    with col5:
        st.metric("🔴 High Risk", stats["high_risk"])

    st.markdown("---")

    # Recent analyses
    st.subheader("📊 Recent Analyses")

    if st.session_state.analysis_history:
        for item in reversed(st.session_state.analysis_history[-10:]):
            result = item["result"]
            risk_class = f"risk-{result.risk_level.lower()}"

            col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
            with col_a:
                st.text(result.email_snippet[:60])
            with col_b:
                st.markdown(f"**{result.prediction}**")
            with col_c:
                st.markdown(f"`{result.confidence:.1%}`")
            with col_d:
                st.markdown(
                    f'<span class="risk-badge {risk_class}">{result.risk_level}</span>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No analyses yet. Go to **🔍 Analyze Email** to start.")

    # System info
    st.markdown("---")
    st.subheader("ℹ️ System Information")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        **Classification Categories:**
        - 🟢 **LEGITIMATE** — Normal, non-malicious email
        - 🟠 **PHISHING** — Social engineering / deception attempt
        - 🔴 **MALICIOUS** — Harmful payload / malware delivery
        """)
    with col_info2:
        st.markdown("""
        **Risk Levels:**
        - 🟢 **LOW** — Minimal concern
        - 🟡 **MEDIUM** — Review recommended
        - 🟠 **HIGH** — Significant risk indicators
        - 🔴 **CRITICAL** — Strong threat indicators
        """)


# ============================================
# PAGE: Analyze Email
# ============================================
elif page == "🔍 Analyze Email":
    render_header()
    st.subheader("📧 Email Analysis")

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
            height=200,
            max_chars=MAX_EMAIL_LENGTH,
            placeholder="Paste the email content here...",
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Email",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not body and not subject:
            st.error("Please provide an email subject or body for analysis.")
        else:
            with st.spinner("Analyzing email..."):
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
            st.markdown("---")
            st.subheader("📊 Analysis Result")

            # Prediction card
            pred_class = f"result-{result.prediction.lower()}"
            st.markdown(
                f'<div class="result-card {pred_class}">'
                f'<h2>Classification: {result.prediction}</h2>'
                f'<p>Confidence: <strong>{result.confidence:.1%}</strong></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Risk and probability columns
            col_risk, col_prob = st.columns(2)

            with col_risk:
                st.markdown("### Risk Assessment")
                risk_class = f"risk-{result.risk_level.lower()}"
                st.markdown(
                    f'<span class="risk-badge {risk_class}">'
                    f'{result.risk_level} RISK</span>',
                    unsafe_allow_html=True,
                )
                st.progress(result.risk_score)
                st.caption(f"Risk Score: {result.risk_score:.3f}")
                st.caption(
                    "⚠️ This is an application-level risk assessment, "
                    "not an objectively validated security rating."
                )

            with col_prob:
                st.markdown("### Class Probabilities")
                for cls, prob in result.probabilities.items():
                    st.markdown(f"**{cls}:** {prob:.1%}")
                    st.progress(prob)

            # Security Indicators
            if result.detected_indicators:
                st.markdown("### 🔒 Detected Security Indicators")
                for indicator in result.detected_indicators:
                    st.markdown(
                        f'<div class="indicator-item">⚠️ {indicator}</div>',
                        unsafe_allow_html=True,
                    )

            # Explanation
            st.markdown("### 📝 Explanation")
            st.info(result.explanation)

            # Recommendation
            st.markdown("### 💡 Recommendation")
            if result.prediction == "LEGITIMATE":
                st.success(result.recommendation)
            elif result.prediction == "PHISHING":
                st.warning(result.recommendation)
            else:
                st.error(result.recommendation)


# ============================================
# PAGE: History
# ============================================
elif page == "📋 History":
    render_header()
    st.subheader("📋 Analysis History")

    if not st.session_state.analysis_history:
        st.info("No analysis history yet.")
    else:
        # Clear button
        if st.button("🗑️ Clear History"):
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

            with st.expander(
                f"{timestamp} | {result.prediction} | "
                f"{result.confidence:.1%} | {result.risk_level}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Prediction:** {result.prediction}")
                    st.markdown(f"**Confidence:** {result.confidence:.1%}")
                with col2:
                    st.markdown(f"**Risk Level:** {result.risk_level}")
                    st.markdown(f"**Risk Score:** {result.risk_score:.3f}")
                with col3:
                    st.markdown(f"**Indicators:** {result.indicator_count}")

                if result.detected_indicators:
                    st.markdown("**Detected Indicators:**")
                    for ind in result.detected_indicators:
                        st.markdown(f"- {ind}")

                st.markdown(f"**Snippet:** {result.email_snippet}")


# ============================================
# PAGE: About
# ============================================
elif page == "ℹ️ About":
    render_header()

    st.markdown("""
    ## About PhishGuard AI

    **PhishGuard AI** is an NLP-based email security analysis system developed as an
    AI/ML capstone project. It combines Natural Language Processing with cybersecurity
    domain knowledge to classify emails and assess security risks.

    ### How It Works

    ```
    Email Input
        ↓
    Text Preprocessing (HTML removal, normalization, tokenization)
        ↓
    Feature Extraction (TF-IDF + Cybersecurity Features)
        ↓
    Machine Learning Classification
        ↓
    Risk Assessment + Explainable Result
    ```

    ### Classification Categories

    | Category | Description |
    |---|---|
    | **LEGITIMATE** | Normal, non-malicious email |
    | **PHISHING** | Social engineering / deception attempt |
    | **MALICIOUS** | Harmful payload / malware delivery |

    ### Important Disclaimers

    - This is a **research/educational capstone prototype**
    - It is NOT a replacement for professional email security products
    - Detection is based on learned patterns from training data and may not
      generalize to all real-world scenarios
    - False positives and false negatives are possible
    - No URLs are visited and no files are executed during analysis

    ### Technologies

    - **NLP:** TF-IDF, tokenization, text preprocessing
    - **ML:** Logistic Regression, Naive Bayes, Linear SVM
    - **Security:** Static URL analysis, keyword-based indicators
    - **Web:** Streamlit
    - **Python:** scikit-learn, pandas, NLTK

    ---

    **Developer:** Muhammad Haris
    **Program:** AI/ML Training Program — KPITB
    """)
