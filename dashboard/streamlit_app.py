"""
CreditIQ — Risk Assessment Console

The dashboard is a client of the CreditIQ FastAPI service.
It does not load the model or access PostgreSQL directly.

Flow:
    Streamlit
        ↓
    FastAPI
        ↓
    CreditIQ Model / Feature Store
"""

import math
import textwrap

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

COLORS = {
    "ink": "#10131C",
    "ink_muted": "#5B6472",
    "surface": "#FFFFFF",
    "app_bg": "#F3F5F8",
    "border": "#E2E5EA",
    "navy": "#16324F",
    "navy_dark": "#0F2438",
    "safe": "#1E7A4C",
    "safe_bg": "#E7F5EC",
    "risk": "#B3261E",
    "risk_bg": "#FBEAEA",
    "amber": "#B9770E",
}

DECISION_STYLES = {
    "APPROVE": {"label": "APPROVED", "glyph": "✓", "fg": COLORS["safe"], "bg": COLORS["safe_bg"]},
    "APPROVED": {"label": "APPROVED", "glyph": "✓", "fg": COLORS["safe"], "bg": COLORS["safe_bg"]},
    "DECLINE": {"label": "DECLINED", "glyph": "✕", "fg": COLORS["risk"], "bg": COLORS["risk_bg"]},
    "DECLINED": {"label": "DECLINED", "glyph": "✕", "fg": COLORS["risk"], "bg": COLORS["risk_bg"]},
    "REVIEW": {"label": "REFER TO REVIEW", "glyph": "!", "fg": COLORS["amber"], "bg": "#FBF2E3"},
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreditIQ — Risk Assessment Console",
    page_icon="◆",
    layout="centered",
)


# ============================================================
# HTML RENDER HELPER
# ------------------------------------------------------------
# Streamlit's markdown parser treats any line indented 4+ spaces
# as a literal code block, which silently renders raw HTML/CSS
# as plain text. textwrap.dedent() strips the common leading
# whitespace that Python's own indentation otherwise adds to
# every multi-line f-string, so tags always parse as HTML
# instead of printing as text. Every HTML/CSS block in this
# file is routed through this one function.
# ============================================================

def render_html(content: str) -> None:
    """
    Renders raw HTML/CSS safely inside Streamlit.

    Two Markdown quirks otherwise cause tags to print as literal text:
    1. Any line indented 4+ spaces is treated as a code block —
       Python's own indentation adds this to every multi-line
       f-string, so textwrap.dedent() strips the common prefix first.
    2. A blank line inside the block can end Markdown's "raw HTML"
       passthrough early, causing whatever follows to be re-parsed
       as an ordinary (non-HTML) paragraph. Blank lines are dropped
       entirely so a whole block is always passed through as one
       uninterrupted chunk of HTML.
    """
    dedented = textwrap.dedent(content).strip()
    collapsed = "\n".join(line for line in dedented.splitlines() if line.strip() != "")
    st.markdown(collapsed, unsafe_allow_html=True)


def humanize_feature_name(name: str) -> str:
    """Turns a raw feature/column name into a readable label."""
    cleaned = name
    for suffix in ("_woe", "_WOE", "_Woe"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.replace("_", " ").strip().upper()


# ============================================================
# GLOBAL STYLE
# ============================================================

render_html(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {COLORS["ink"]};
    }}

    .stApp {{
        background-color: {COLORS["app_bg"]};
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    .block-container {{
        max-width: 760px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }}

    /* ---------------- Native bordered containers -> cards ---------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(16, 19, 28, 0.04);
        margin-bottom: 22px;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding: 6px 4px;
    }}

    .card-title {{
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {COLORS["ink_muted"]};
        margin: 0 0 18px 0;
    }}

    /* ---------------- Masthead ---------------- */
    .ciq-masthead {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 14px;
        border-bottom: 1px solid {COLORS["border"]};
        margin-bottom: 28px;
    }}
    .ciq-wordmark {{
        display: flex;
        align-items: baseline;
        gap: 10px;
    }}
    .ciq-wordmark .mark {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
        color: {COLORS["navy"]};
        border: 1.5px solid {COLORS["navy"]};
        border-radius: 6px;
        padding: 2px 7px;
    }}
    .ciq-wordmark .eyebrow {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        color: {COLORS["ink_muted"]};
        text-transform: uppercase;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        padding: 5px 11px;
        border-radius: 999px;
        white-space: nowrap;
    }}
    .status-pill .dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }}
    .status-online {{ background: {COLORS["safe_bg"]}; color: {COLORS["safe"]}; }}
    .status-online .dot {{ background: {COLORS["safe"]}; }}
    .status-offline {{ background: {COLORS["risk_bg"]}; color: {COLORS["risk"]}; }}
    .status-offline .dot {{ background: {COLORS["risk"]}; }}
    .status-degraded {{ background: #FBF2E3; color: {COLORS["amber"]}; }}
    .status-degraded .dot {{ background: {COLORS["amber"]}; }}

    /* ---------------- Decision card ---------------- */
    .decision-badge {{
        display: inline-flex;
        align-items: center;
        gap: 9px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.05em;
        padding: 12px 22px;
        border-radius: 10px;
    }}
    .decision-badge .glyph {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: currentColor;
        font-size: 0.8rem;
    }}
    .decision-badge .glyph span {{
        color: white;
    }}
    .threshold-note {{
        font-size: 0.78rem;
        color: {COLORS["ink_muted"]};
        margin-top: 10px;
        line-height: 1.5;
    }}

    /* ---------------- Data strip ---------------- */
    .data-strip {{
        display: flex;
        border-top: 1px solid {COLORS["border"]};
        margin-top: 22px;
        padding-top: 18px;
    }}
    .data-cell {{
        flex: 1;
        padding: 0 18px;
        border-left: 1px solid {COLORS["border"]};
    }}
    .data-cell:first-child {{ border-left: none; padding-left: 0; }}
    .data-cell .label {{
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLORS["ink_muted"]};
        margin-bottom: 5px;
    }}
    .data-cell .value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.28rem;
        font-weight: 600;
        color: {COLORS["ink"]};
        white-space: nowrap;
    }}

    /* ---------------- Ledger bars ---------------- */
    .ledger-column-title {{
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 14px;
    }}
    .ledger-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 13px;
    }}
    .ledger-label {{
        flex: 0 0 40%;
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        color: {COLORS["ink"]};
        line-height: 1.25;
    }}
    .ledger-track {{
        flex: 1;
        height: 9px;
        background: {COLORS["app_bg"]};
        border-radius: 5px;
        overflow: hidden;
    }}
    .ledger-fill {{
        height: 100%;
        border-radius: 5px;
    }}
    .ledger-value {{
        flex: 0 0 58px;
        text-align: right;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: {COLORS["ink_muted"]};
    }}
    .ledger-footnote {{
        font-size: 0.72rem;
        color: {COLORS["ink_muted"]};
        margin-top: 6px;
        padding-top: 14px;
        border-top: 1px solid {COLORS["border"]};
    }}

    /* ---------------- Streamlit control overrides ---------------- */
    div[data-testid="stNumberInput"] input {{
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 8px !important;
        border: 1px solid {COLORS["border"]} !important;
    }}
    .stButton > button {{
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.6rem 1.1rem;
        border: 1px solid {COLORS["navy"]};
        transition: background 0.15s ease, color 0.15s ease;
    }}
    .stButton > button[kind="primary"] {{
        background: {COLORS["navy"]};
        color: #FFFFFF;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {COLORS["navy_dark"]};
        border-color: {COLORS["navy_dark"]};
        color: #FFFFFF;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent;
        color: {COLORS["navy"]};
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: {COLORS["app_bg"]};
        color: {COLORS["navy_dark"]};
    }}
    .footnote {{
        font-size: 0.74rem;
        color: {COLORS["ink_muted"]};
        margin-top: 6px;
    }}
    </style>
    """
)


# ============================================================
# GAUGE
# ============================================================

def render_gauge(probability: float, threshold: float) -> str:
    """
    Builds an inline SVG semicircular risk gauge.
    Safe zone spans [0, threshold], risk zone spans [threshold, 1].
    A tick marks the exact decision threshold; a needle marks the
    applicant's calibrated probability of default.
    """
    probability = min(max(probability, 0.0), 1.0)
    threshold = min(max(threshold, 0.0), 1.0)

    cx, cy, r = 110, 106, 82

    def point(fraction: float, radius: float) -> tuple:
        angle = math.pi * (1 - fraction)  # 0 -> 180deg (left), 1 -> 0deg (right)
        return (
            cx + radius * math.cos(angle),
            cy - radius * math.sin(angle),
        )

    def arc_path(f_start: float, f_end: float, radius: float) -> str:
        x1, y1 = point(f_start, radius)
        x2, y2 = point(f_end, radius)
        return f"M {x1:.2f} {y1:.2f} A {radius} {radius} 0 0 1 {x2:.2f} {y2:.2f}"

    safe_path = arc_path(0.0, threshold, r)
    risk_path = arc_path(threshold, 1.0, r)

    tick_inner = point(threshold, r - 13)
    tick_outer = point(threshold, r + 7)

    needle_tip = point(probability, r - 20)
    needle_color = COLORS["risk"] if probability >= threshold else COLORS["navy"]

    return f"""
    <svg viewBox="0 0 220 126" width="100%" height="150" role="img" aria-label="Risk gauge">
        <path d="{safe_path}" fill="none" stroke="{COLORS['safe']}" stroke-width="13" opacity="0.55" />
        <path d="{risk_path}" fill="none" stroke="{COLORS['risk']}" stroke-width="13" opacity="0.45" />
        <line x1="{tick_inner[0]:.2f}" y1="{tick_inner[1]:.2f}" x2="{tick_outer[0]:.2f}" y2="{tick_outer[1]:.2f}"
              stroke="{COLORS['ink']}" stroke-width="2.5" />
        <circle cx="{cx}" cy="{cy}" r="5" fill="{COLORS['ink']}" />
        <line x1="{cx}" y1="{cy}" x2="{needle_tip[0]:.2f}" y2="{needle_tip[1]:.2f}"
              stroke="{needle_color}" stroke-width="3.5" stroke-linecap="round" />
        <text x="{cx}" y="{cy + 32}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
              font-size="21" font-weight="600" fill="{COLORS['ink']}">{probability:.2%}</text>
        <text x="{cx}" y="{cy + 48}" text-anchor="middle" font-family="IBM Plex Sans, sans-serif"
              font-size="9" letter-spacing="1.3" fill="{COLORS['ink_muted']}">PROBABILITY OF DEFAULT</text>
    </svg>
    """


def render_ledger_bars(factors: list, max_value: float, fill_color: str) -> str:
    if not factors:
        return "<div class='footnote'>No contributing factors returned.</div>"

    rows = ""
    for factor in factors:
        magnitude = abs(factor.get("contribution", 0))
        width_pct = 6 if max_value == 0 else max(6, (magnitude / max_value) * 100)
        label = humanize_feature_name(str(factor.get("feature", "—")))
        rows += (
            '<div class="ledger-row">'
            f'<div class="ledger-label">{label}</div>'
            f'<div class="ledger-track"><div class="ledger-fill" '
            f'style="width:{width_pct:.0f}%; background:{fill_color};"></div></div>'
            f'<div class="ledger-value">{factor.get("contribution", 0):.3f}</div>'
            "</div>"
        )
    return rows


# ============================================================
# MASTHEAD
# ============================================================

status_html = '<span class="status-pill status-offline"><span class="dot"></span>SYSTEM OFFLINE</span>'

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)

    if health_response.status_code == 200:
        health_data = health_response.json()

        if health_data.get("model_loaded") and health_data.get("database_connected"):
            status_html = '<span class="status-pill status-online"><span class="dot"></span>SYSTEM ONLINE</span>'
        else:
            status_html = '<span class="status-pill status-degraded"><span class="dot"></span>DEPENDENCY DEGRADED</span>'
    else:
        status_html = '<span class="status-pill status-degraded"><span class="dot"></span>UNEXPECTED STATUS</span>'

except requests.RequestException:
    status_html = '<span class="status-pill status-offline"><span class="dot"></span>API UNREACHABLE</span>'

render_html(
    f"""
    <div class="ciq-masthead">
        <div class="ciq-wordmark">
            <span class="mark">CR·IQ</span>
            <span class="eyebrow">Risk Assessment Console</span>
        </div>
        {status_html}
    </div>
    """
)

if "API UNREACHABLE" in status_html:
    render_html(
        "<div class='footnote'>Can't reach the CreditIQ API. "
        "Confirm the service is running on port 8000.</div>"
    )


# ============================================================
# APPLICANT LOOKUP
# ============================================================

with st.container(border=True):
    render_html('<div class="card-title">Applicant Lookup</div>')

    col_input, col_button = st.columns([3, 1.4])

    with col_input:
        applicant_id = st.number_input(
            "Applicant ID",
            min_value=1,
            step=1,
            value=447009,
            label_visibility="collapsed",
        )

    with col_button:
        score_clicked = st.button("Assess Applicant →", type="primary", use_container_width=True)

if score_clicked:
    try:
        response = requests.post(
            f"{API_URL}/score",
            json={"applicant_id": int(applicant_id)},
            timeout=10,
        )

        if response.status_code == 200:
            st.session_state["score_result"] = response.json()
            st.session_state.pop("explanation", None)
        elif response.status_code == 404:
            st.error(f"Applicant {applicant_id} was not found. Check the ID and try again.")
            st.session_state.pop("score_result", None)
        else:
            st.error(f"Scoring failed: {response.text}")

    except requests.RequestException as error:
        st.error(f"Can't reach the CreditIQ API: {error}")


# ============================================================
# DECISION CARD
# ============================================================

if "score_result" in st.session_state:

    result = st.session_state["score_result"]
    probability = result["probability_of_default"]
    threshold = result["threshold"]
    decision_key = str(result["decision"]).upper()
    style = DECISION_STYLES.get(
        decision_key,
        {"label": decision_key, "glyph": "•", "fg": COLORS["ink_muted"], "bg": COLORS["app_bg"]},
    )

    with st.container(border=True):
        render_html(f'<div class="card-title">Credit Decision — Applicant {int(applicant_id)}</div>')

        col_gauge, col_verdict = st.columns([1, 1.1])

        with col_gauge:
            render_html(render_gauge(probability, threshold))

        with col_verdict:
            render_html(
                f"""
                <div class="decision-badge" style="color:{style['fg']}; background:{style['bg']};">
                    <span class="glyph"><span>{style['glyph']}</span></span>{style['label']}
                </div>
                <div class="threshold-note">
                    Decision reflects a frozen threshold of {threshold:.2%},
                    selected on validation data prior to deployment.
                </div>
                """
            )

        render_html(
            f"""
            <div class="data-strip">
                <div class="data-cell">
                    <div class="label">Probability of Default</div>
                    <div class="value">{probability:.2%}</div>
                </div>
                <div class="data-cell">
                    <div class="label">Decision Threshold</div>
                    <div class="value">{threshold:.2%}</div>
                </div>
                <div class="data-cell">
                    <div class="label">Model Version</div>
                    <div class="value" style="font-size:0.98rem;">{result['model_version']}</div>
                </div>
            </div>
            """
        )

    explain_clicked = st.button("Explain This Decision")

    if explain_clicked:
        try:
            response = requests.post(
                f"{API_URL}/explain",
                json={"applicant_id": int(applicant_id)},
                timeout=10,
            )

            if response.status_code == 200:
                st.session_state["explanation"] = response.json()
            else:
                st.error(f"Explanation failed: {response.text}")

        except requests.RequestException as error:
            st.error(f"Can't reach the CreditIQ API: {error}")


# ============================================================
# EXPLANATION — RISK LEDGER
# ============================================================

if "explanation" in st.session_state:

    explanation = st.session_state["explanation"]
    risk_factors = explanation.get("top_risk_factors", [])
    protective_factors = explanation.get("top_protective_factors", [])

    all_magnitudes = [abs(f.get("contribution", 0)) for f in risk_factors + protective_factors]
    max_magnitude = max(all_magnitudes) if all_magnitudes else 0

    with st.container(border=True):
        render_html('<div class="card-title">Risk Ledger</div>')

        col_risk, col_protect = st.columns(2)

        with col_risk:
            render_html(
                f'<div class="ledger-column-title" style="color:{COLORS["risk"]};">↑ Increases Risk</div>'
                + render_ledger_bars(risk_factors, max_magnitude, COLORS["risk"])
            )

        with col_protect:
            render_html(
                f'<div class="ledger-column-title" style="color:{COLORS["safe"]};">↓ Reduces Risk</div>'
                + render_ledger_bars(protective_factors, max_magnitude, COLORS["safe"])
            )

        render_html(
            f"""
            <div class="ledger-footnote">
                Model: {explanation.get('model_version', '—')} ·
                Contributions are SHAP values on the underlying model output.
            </div>
            """
        )