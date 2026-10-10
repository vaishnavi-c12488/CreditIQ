"""
CreditIQ Underwriting Console

Streamlit client for the CreditIQ FastAPI service. It never touches the model
or PostgreSQL directly.

    Streamlit -> FastAPI (/health, /score, /explain) -> Model / Feature Store
"""

import json
import os
import textwrap

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

INK, OK, BAD, WARN, TEAL = "#0C1A24", "#0F8A5F", "#C73A3A", "#BE7C0C", "#0B5F6B"

DECISIONS = {
    "APPROVE": ("Approve", OK),
    "APPROVED": ("Approve", OK),
    "DECLINE": ("Decline", BAD),
    "DECLINED": ("Decline", BAD),
    "REVIEW": ("Refer to underwriter", WARN),
}

st.set_page_config(
    page_title="CreditIQ Underwriting Console",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_html(content: str) -> None:
    """Dedent and drop blank lines so Streamlit's Markdown never shows tags as text."""
    text = textwrap.dedent(content).strip()
    st.markdown("\n".join(l for l in text.splitlines() if l.strip()), unsafe_allow_html=True)


def humanize(name: str) -> str:
    for suffix in ("_woe", "_WOE", "_Woe"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", " ").strip().title()


def call_api(path: str, payload: dict):
    """Returns (data, error_message)."""
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    except requests.RequestException as exc:
        return None, f"Can't reach the CreditIQ API at {API_URL}. Start the service and try again. ({exc})"
    if r.status_code == 200:
        return r.json(), None
    if r.status_code == 404:
        return None, f"Applicant {payload['applicant_id']} wasn't found. Check the ID and try again."
    return None, f"The API returned an error ({r.status_code}): {r.text}"


# ------------------------------------------------------------------ styles
render_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');
:root{--ink:#0C1A24;--mist:#EAF0F2;--line:#D5DEE2;--muted:#55666F;--teal:#0B5F6B;--deep:#072B36;--ok:#0F8A5F;--bad:#C73A3A;--warn:#BE7C0C;}
html,body,[class*="css"]{font-family:'Figtree',sans-serif;color:var(--ink);}
.stApp{background:var(--mist);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{max-width:1360px;padding:1.25rem 3vw 3rem;}

/* hero */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:28px 40px 56px;color:#fff;
  background:linear-gradient(120deg,var(--deep) 0%,var(--teal) 100%);margin-bottom:18px;}
.hero::after{content:"";position:absolute;left:0;right:0;bottom:0;height:16px;
  background:repeating-linear-gradient(90deg,rgba(255,255,255,.22) 0 1px,transparent 1px 24px);}
.nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:44px;}
.brand{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.35rem;letter-spacing:-.01em;}
.brand i{font-style:normal;opacity:.65;font-weight:600;margin-left:8px;font-size:1rem;}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;font-size:.82rem;font-weight:600;background:rgba(255,255,255,.14);}
.pill b{width:8px;height:8px;border-radius:50%;display:inline-block;}
.hero h1{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:clamp(2rem,4.2vw,3.6rem);line-height:1.04;letter-spacing:-.025em;margin:0 0 14px;max-width:16ch;color:#fff;}
.hero p{max-width:58ch;font-size:1.05rem;line-height:1.55;opacity:.86;margin:0;}

/* containers */
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line)!important;border-radius:18px!important;box-shadow:0 1px 2px rgba(12,26,36,.04);}
div[data-testid="stForm"]{border:none;padding:0;}
div[data-testid="stNumberInput"] input{font-size:1.15rem;font-weight:600;border-radius:12px!important;}
.stButton>button,.stDownloadButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:700;padding:.65rem 1.2rem;border:1.5px solid var(--teal);}
div[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"],.stButton>button[kind="primary"]{background:var(--teal);color:#fff;}
div[data-testid="stFormSubmitButton"]>button:hover{background:var(--deep);border-color:var(--deep);color:#fff;}

/* verdict */
.sec{font-family:'Bricolage Grotesque',sans-serif;font-weight:600;font-size:1.05rem;margin:0 0 4px;}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 16px;line-height:1.5;}
.verdict{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:clamp(2.2rem,4vw,3.4rem);letter-spacing:-.025em;line-height:1;margin:2px 0 8px;}
.stats{display:flex;border-top:1px solid var(--line);margin-top:6px;padding-top:16px;}
.stat{flex:1;padding:0 18px;border-left:1px solid var(--line);}
.stat:first-child{border-left:none;padding-left:0;}
.stat span{display:block;font-size:.8rem;color:var(--muted);margin-bottom:4px;}
.stat b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.45rem;font-weight:800;}
.note{font-size:.85rem;color:var(--muted);line-height:1.5;margin-top:10px;}
.sim{border-radius:12px;background:var(--mist);padding:12px 16px;font-size:.92rem;font-weight:500;margin-top:6px;}

/* risk spectrum */
.spec{position:relative;height:150px;margin:34px 10px 4px;}
.spec-track{position:absolute;top:52px;left:0;right:0;height:16px;border-radius:8px;background:linear-gradient(90deg,#F2CFCA,#E59A92);}
.spec-ok{position:absolute;left:0;top:0;bottom:0;border-radius:8px 0 0 8px;background:linear-gradient(90deg,#C4EAD8,#7CCBA8);}
.cut{position:absolute;top:30px;height:60px;border-left:2px solid var(--ink);}
.cut.test{top:4px;height:86px;border-left:2px dashed var(--teal);}
.cut span{position:absolute;bottom:100%;left:0;transform:translateX(-50%);padding-bottom:3px;font-size:.78rem;font-weight:700;white-space:nowrap;}
.cut.test span{color:var(--teal);}
.pin{position:absolute;top:44px;width:32px;height:32px;margin-left:-16px;border-radius:50%;background:var(--c);border:4px solid #fff;
  box-shadow:0 2px 10px rgba(12,26,36,.35);animation:slideIn 1.1s cubic-bezier(.2,.8,.2,1);}
.pin b{position:absolute;top:100%;left:50%;transform:translateX(-50%);margin-top:6px;font-family:'Bricolage Grotesque',sans-serif;font-size:1.05rem;color:var(--c);white-space:nowrap;}
.ticks{position:absolute;top:124px;left:0;right:0;}
.ticks span{position:absolute;transform:translateX(-50%);font-size:.74rem;color:var(--muted);}
@keyframes slideIn{from{left:0;}}
@media (prefers-reduced-motion:reduce){.pin{animation:none;}}

/* drivers */
.drv-head{display:flex;justify-content:space-between;font-size:.8rem;font-weight:700;margin:2px 0 14px;}
.drv-head span:first-child{color:var(--ok);} .drv-head span:last-child{color:var(--bad);}
.drv{display:grid;grid-template-columns:38% 1fr 62px;gap:12px;align-items:center;margin-bottom:14px;}
.drv-name{font-size:.88rem;font-weight:600;line-height:1.25;}
.drv-bar{position:relative;height:12px;background:var(--mist);border-radius:6px;}
.drv-bar::before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#9AA9B1;}
.drv-bar i{position:absolute;top:0;bottom:0;}
.drv-bar i.up{left:50%;background:var(--bad);border-radius:0 6px 6px 0;}
.drv-bar i.down{right:50%;background:var(--ok);border-radius:6px 0 0 6px;}
.drv-val{text-align:right;font-size:.82rem;font-weight:600;color:var(--muted);}
.empty{text-align:center;padding:44px 12px;color:var(--muted);font-size:1rem;}
.empty b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.4rem;color:var(--ink);margin-bottom:6px;}
</style>
"""
)


# --------------------------------------------------------------- components
def spectrum_html(p: float, thr: float, test: float, color: str) -> str:
    vmax = min(1.0, max(0.5, max(p, thr, test) * 1.3))
    pos = lambda v: min(max(v / vmax, 0.0), 1.0) * 100
    test_cut = ""
    if abs(test - thr) > 1e-9:
        test_cut = f'<div class="cut test" style="left:{pos(test):.2f}%"><span>Test cutoff {test:.0%}</span></div>'
    ticks = "".join(f'<span style="left:{i * 25}%">{vmax * i / 4:.0%}</span>' for i in range(5))
    return f"""
    <div class="spec">
      <div class="spec-track"><div class="spec-ok" style="width:{pos(thr):.2f}%"></div></div>
      <div class="cut" style="left:{pos(thr):.2f}%"><span>Approved cutoff {thr:.0%}</span></div>
      {test_cut}
      <div class="pin" style="left:{pos(p):.2f}%;--c:{color}"><b>{p:.1%}</b></div>
      <div class="ticks">{ticks}</div>
    </div>
    """


def drivers_html(risk: list, protective: list) -> str:
    items = [(f, 1) for f in risk] + [(f, -1) for f in protective]
    if not items:
        return '<div class="note">The API returned no contributing factors for this applicant.</div>'
    items.sort(key=lambda it: -abs(it[0].get("contribution", 0)))
    top = max(abs(f.get("contribution", 0)) for f, _ in items) or 1.0
    rows = ""
    for f, side in items:
        c = abs(f.get("contribution", 0))
        width = max(c / top * 50, 2)
        cls = "up" if side > 0 else "down"
        sign = "+" if side > 0 else "−"
        rows += (
            f'<div class="drv"><div class="drv-name">{humanize(str(f.get("feature", "Unknown")))}</div>'
            f'<div class="drv-bar"><i class="{cls}" style="width:{width:.1f}%"></i></div>'
            f'<div class="drv-val">{sign}{c:.3f}</div></div>'
        )
    return f'<div class="drv-head"><span>← Lowers risk</span><span>Raises risk →</span></div>{rows}'


# ------------------------------------------------------------------- state
st.session_state.setdefault("history", [])
st.session_state.setdefault("sim_cutoff", 20)

# ------------------------------------------------------------------- hero
dot, status = "#F2B544", "Checking API"
try:
    h = requests.get(f"{API_URL}/health", timeout=3)
    if h.status_code == 200:
        d = h.json()
        ready = d.get("model_loaded") and d.get("database_connected")
        dot, status = ("#4ADE9B", "Model and database ready") if ready else ("#F2B544", "A dependency is degraded")
    else:
        dot, status = "#F2B544", f"API returned {h.status_code}"
except requests.RequestException:
    dot, status = "#FF7B72", "API unreachable"

render_html(
    f"""
    <div class="hero">
      <div class="nav">
        <div class="brand">CreditIQ<i>Underwriting console</i></div>
        <div class="pill"><b style="background:{dot}"></b>{status}</div>
      </div>
      <h1>Know the risk before you lend.</h1>
      <p>Enter an applicant ID to see the calibrated probability of default, the decision at the approved cutoff,
      and the factors that moved the score.</p>
    </div>
    """
)

# ------------------------------------------------------------------ lookup
with st.container(border=True):
    with st.form("lookup", border=False):
        c_in, c_btn = st.columns([4, 1.2], vertical_alignment="bottom")
        applicant_id = c_in.number_input("Applicant ID", min_value=1, step=1, value=447009)
        submitted = c_btn.form_submit_button("Assess applicant", type="primary", use_container_width=True)

if submitted:
    aid = int(applicant_id)
    with st.spinner("Scoring applicant..."):
        score, err = call_api("/score", {"applicant_id": aid})
        explain, _ = call_api("/explain", {"applicant_id": aid}) if not err else (None, None)
    if err:
        st.session_state.pop("result", None)
        st.error(err)
    else:
        st.session_state["result"] = {"id": aid, "score": score, "explain": explain or {}}
        st.session_state["sim_cutoff"] = int(min(60, max(1, round(score["threshold"] * 100))))
        label = DECISIONS.get(str(score["decision"]).upper(), (str(score["decision"]), INK))[0]
        entry = {
            "Applicant": aid,
            "Default probability": f"{score['probability_of_default']:.2%}",
            "Decision": label,
        }
        if not st.session_state["history"] or st.session_state["history"][-1] != entry:
            st.session_state["history"].append(entry)

# ----------------------------------------------------------------- results
res = st.session_state.get("result")

if not res:
    with st.container(border=True):
        render_html(
            '<div class="empty"><b>No applicant assessed yet</b>'
            "Enter an ID above and select Assess applicant to see a decision.</div>"
        )
else:
    score, explain = res["score"], res["explain"]
    p, thr = score["probability_of_default"], score["threshold"]
    label, color = DECISIONS.get(str(score["decision"]).upper(), (str(score["decision"]), INK))
    test = st.session_state["sim_cutoff"] / 100

    left, right = st.columns([1.45, 1], gap="medium")

    with left:
        with st.container(border=True):
            render_html(f'<div class="sec">Applicant {res["id"]}</div>')
            render_html(
                f"""
                <div class="verdict" style="color:{color}">{label}</div>
                <div class="sub">Probability of default is {p:.1%}, {"below" if p < thr else "at or above"}
                the {thr:.0%} cutoff set before deployment.</div>
                """
            )
            render_html(spectrum_html(p, thr, test, color))
            render_html(
                f"""
                <div class="stats">
                  <div class="stat"><span>Probability of default</span><b>{p:.2%}</b></div>
                  <div class="stat"><span>Approved cutoff</span><b>{thr:.2%}</b></div>
                  <div class="stat"><span>Model version</span><b style="font-size:1.05rem">{score["model_version"]}</b></div>
                </div>
                """
            )

        with st.container(border=True):
            render_html(
                '<div class="sec">Try a different cutoff</div>'
                '<div class="sub">Drag to see how a stricter or looser policy would treat this applicant.</div>'
            )
            st.slider("Test cutoff (%)", 1, 60, key="sim_cutoff", label_visibility="collapsed")
            test = st.session_state["sim_cutoff"] / 100
            outcome = "declined" if p >= test else "approved"
            if abs(test - thr) < 1e-9:
                msg = f"This matches the deployed policy. The applicant is {outcome}."
            else:
                msg = f"At a {test:.0%} cutoff, this applicant would be {outcome}."
            render_html(f'<div class="sim">{msg}</div>')
            render_html(
                f'<div class="note">This is a what-if only. The decision above always uses the frozen {thr:.0%} cutoff.</div>'
            )

    with right:
        with st.container(border=True):
            render_html(
                '<div class="sec">What drove this score</div>'
                '<div class="sub">Each bar is a SHAP contribution to the model output.</div>'
            )
            render_html(
                drivers_html(explain.get("top_risk_factors", []), explain.get("top_protective_factors", []))
            )
            if not explain:
                st.warning("The explanation request failed, so factors aren't shown. Assess the applicant again to retry.")
            st.download_button(
                "Download report (JSON)",
                data=json.dumps({"applicant_id": res["id"], "score": score, "explanation": explain}, indent=2),
                file_name=f"creditiq_{res['id']}.json",
                mime="application/json",
                use_container_width=True,
            )

# ----------------------------------------------------------------- history
if st.session_state["history"]:
    with st.container(border=True):
        render_html('<div class="sec">Recent assessments</div><div class="sub">This session, newest first.</div>')
        st.dataframe(list(reversed(st.session_state["history"]))[:8], hide_index=True, use_container_width=True)
