"""
Visual system for the dashboard: palettes, global CSS and reusable HTML components.

Nothing in this module touches data or models. Colours are defined once per theme
and exposed as CSS variables, so every component (including Streamlit's own
widgets) follows the active theme. Charts read the same palette dictionary.
"""

from __future__ import annotations

import html as _html

import pandas as pd

FONT_DISPLAY = "'Sora', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_BODY = "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
FONT_STACK = FONT_BODY  # used by Plotly figures

_DARK = {
    "scheme": "dark",
    "bg": "#031326",
    "bg_soft": "#06192F",
    "surface": "#0A203A",
    "surface_2": "#0F2A4A",
    "surface_hover": "#132F55",
    "border": "#1A3658",
    "border_strong": "#284B72",
    "text": "#FFFFFF",
    "text_2": "#C5D1E2",
    "muted": "#7F93AF",
    "accent": "#D100D1",
    "accent_soft": "rgba(209, 0, 209, 0.14)",
    "accent_line": "rgba(209, 0, 209, 0.55)",
    "ok": "#3DD6C6",
    "ok_soft": "rgba(61, 214, 198, 0.12)",
    "row_alt": "rgba(255, 255, 255, 0.018)",
    "shadow_card": "0 1px 0 rgba(255,255,255,0.03) inset, 0 16px 40px rgba(0,0,0,0.28)",
    "shadow_lift": "0 22px 50px rgba(0,0,0,0.38)",
    "grid": "rgba(143, 168, 200, 0.13)",
    # chart semantics
    "benign": "#3DD6C6",
    "malignant": "#FF6B7A",
    "series_a": "#5BB8FF",
    "series_b": "#D100D1",
    "cm_scale": [[0.0, "#0C2544"], [1.0, "#2F76B4"]],
    "cm_text_lo": "#FFFFFF",
    "cm_text_hi": "#FFFFFF",
    "corr_scale": [[0.0, "#2F6FD0"], [0.5, "#17304F"], [1.0, "#D100D1"]],
    "models": {
        "KNN": "#5BB8FF", "Decision Tree": "#F5B85B", "Random Forest": "#3DD6A0",
        "ANN (MLP)": "#D100D1", "SVM": "#9C8BFF",
    },
    "metrics": {
        "Accuracy": "#7FA7FF", "Precision": "#3DD6C6", "Recall": "#D100D1",
        "F1-score": "#F5B85B", "ROC-AUC": "#9C8BFF",
    },
}

_LIGHT = {
    "scheme": "light",
    "bg": "#F3F5F9",
    "bg_soft": "#EBEFF6",
    "surface": "#FFFFFF",
    "surface_2": "#E9EDF4",
    "surface_hover": "#F4F7FB",
    "border": "#DCE2EC",
    "border_strong": "#C6CFDD",
    "text": "#15243A",
    "text_2": "#33465F",
    "muted": "#65738A",
    "accent": "#A900A9",
    "accent_soft": "rgba(169, 0, 169, 0.09)",
    "accent_line": "rgba(169, 0, 169, 0.38)",
    "ok": "#0E9F8E",
    "ok_soft": "rgba(14, 159, 142, 0.10)",
    "row_alt": "#F7F9FC",
    "shadow_card": "0 1px 2px rgba(21,36,58,0.04), 0 10px 26px rgba(21,36,58,0.06)",
    "shadow_lift": "0 18px 40px rgba(21,36,58,0.12)",
    "grid": "rgba(21, 36, 58, 0.08)",
    "benign": "#0E9F8E",
    "malignant": "#E0455F",
    "series_a": "#2B7BC4",
    "series_b": "#A900A9",
    "cm_scale": [[0.0, "#EEF3FA"], [1.0, "#2B7BC4"]],
    "cm_text_lo": "#15243A",
    "cm_text_hi": "#FFFFFF",
    "corr_scale": [[0.0, "#2B6CD0"], [0.5, "#F1F4F9"], [1.0, "#A900A9"]],
    "models": {
        "KNN": "#2B7BC4", "Decision Tree": "#C98A1E", "Random Forest": "#1E9A6C",
        "ANN (MLP)": "#A900A9", "SVM": "#5B63D6",
    },
    "metrics": {
        "Accuracy": "#4D7FD6", "Precision": "#0E9F8E", "Recall": "#A900A9",
        "F1-score": "#C98A1E", "ROC-AUC": "#5B63D6",
    },
}


def palette(theme: str) -> dict:
    return dict(_LIGHT if theme == "light" else _DARK)


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
def _root_block(p: dict) -> str:
    lines = [f"  --{k.replace('_', '-')}: {v};" for k, v in p.items() if isinstance(v, str) and k != "scheme"]
    lines += [
        f"  --font-display: {FONT_DISPLAY};",
        f"  --font-body: {FONT_BODY};",
        f"  --font-mono: {FONT_MONO};",
    ]
    return ":root {\n" + "\n".join(lines) + f"\n  color-scheme: {p['scheme']};\n}}"


_STATIC_CSS = """

html, body, .stApp, [data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text);
  font-family: var(--font-body);
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1320px; padding: 2.2rem 2.6rem 4rem; }
[data-testid="stHorizontalBlock"] { gap: 1.15rem; }
[data-testid="column"] { min-width: 0; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebarContent"] { padding-top: 1.4rem; }

h1, h2, h3, h4 { font-family: var(--font-display); color: var(--text); letter-spacing: -0.02em; }
[data-testid="stMarkdownContainer"] p { color: var(--text-2); }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
  color: var(--text-2) !important; font-size: 0.82rem !important; font-weight: 500;
}

/* Select, multiselect, number and text inputs */
[data-baseweb="select"] > div, [data-baseweb="input"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] input,
[data-baseweb="input"] input, [data-testid="stNumberInput"] input { color: var(--text) !important; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {
  background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
}
[data-baseweb="menu"] li { color: var(--text) !important; }
[data-baseweb="menu"] li:hover, [data-baseweb="menu"] li[aria-selected="true"] { background: var(--accent-soft) !important; }
[data-baseweb="tag"] { background: var(--accent-soft) !important; border-radius: 8px !important; }
[data-baseweb="tag"] span { color: var(--text) !important; }

/* Sliders and toggles */
[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important; box-shadow: 0 0 0 4px var(--accent-soft) !important;
}
[data-testid="stThumbValue"] { color: var(--text) !important; font-family: var(--font-mono); }
[data-testid="stCheckbox"] label p, [data-testid="stToggle"] label p { color: var(--text) !important; }

/* Tabs */
[data-baseweb="tab-list"] { gap: 4px; background: transparent !important; }
[data-baseweb="tab-border"] { background: var(--border) !important; }
[data-baseweb="tab"] {
  background: transparent !important; color: var(--muted) !important; font-weight: 500;
  padding: .6rem 1rem !important; border-radius: 10px 10px 0 0 !important; transition: color .2s ease, background .2s ease;
}
[data-baseweb="tab"]:hover { color: var(--text) !important; background: var(--accent-soft) !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; font-weight: 600; }
[data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 12px; font-weight: 600; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }

/* Bordered containers become the premium chart / panel cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 20px !important;
  box-shadow: var(--shadow-card);
  padding: 1.1rem 1.25rem .5rem !important;
  transition: border-color .25s ease, box-shadow .25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--accent-line) !important; }
[data-testid="stPlotlyChart"] { background: transparent; }

/* Sidebar navigation: radio options styled as navigation items */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
  width: 100%; margin: 0; padding: .62rem .8rem; border-radius: 12px; border: 1px solid transparent;
  cursor: pointer; transition: background .2s ease, border-color .2s ease, transform .2s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: var(--surface-hover); transform: translateX(3px); }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] [role="radiogroup"] p { color: var(--text-2); font-size: .94rem; font-weight: 500; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
  background: var(--accent-soft); border-color: var(--accent-line); box-shadow: inset 3px 0 0 var(--accent);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: var(--text); font-weight: 600; }

/* ---------- Components ---------- */
.ds-eyebrow { font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
.ds-eyebrow-accent { color: var(--accent); }
.ds-lede { font-size: 1rem; line-height: 1.65; color: var(--text-2); max-width: 720px; margin: 0; }

.ds-hero {
  position: relative; overflow: hidden; border-radius: 24px; padding: 2.5rem 2.6rem;
  border: 1px solid var(--border); margin-bottom: 1.5rem; box-shadow: var(--shadow-card);
  background: linear-gradient(135deg, var(--surface) 0%, var(--bg-soft) 100%);
}
.ds-hero::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(60% 90% at 100% 0%, var(--accent-soft), transparent 70%);
}
.ds-hero > :not(.ds-hero-art) { position: relative; }
.ds-hero-art { position: absolute; right: 1.8rem; top: 1.4rem; width: 280px; height: 170px; color: var(--muted); pointer-events: none; }
.ds-title {
  font-family: var(--font-display); font-size: clamp(1.9rem, 3.2vw, 2.7rem); font-weight: 600;
  line-height: 1.12; letter-spacing: -0.03em; color: var(--text); margin: .75rem 0 .85rem; max-width: 720px;
}
.ds-chip-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.5rem; }
.ds-chip {
  display: inline-flex; align-items: center; gap: .35rem; padding: .34rem .72rem; border-radius: 999px;
  background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2); font-size: .8rem; line-height: 1.2;
}
.ds-chip.mono { font-family: var(--font-mono); font-size: .78rem; }
.ds-chip.accent { border-color: var(--accent-line); background: var(--accent-soft); color: var(--text); }

.ds-page-intro { margin: .1rem 0 1.5rem; }
.ds-page-title { font-family: var(--font-display); font-size: 2.05rem; font-weight: 600; letter-spacing: -0.03em; color: var(--text); margin: .45rem 0 .5rem; }

.ds-section { display: flex; gap: 1rem; align-items: flex-start; margin: 2.2rem 0 1rem; }
.ds-section-index {
  font-family: var(--font-mono); font-size: .76rem; color: var(--accent); background: var(--accent-soft);
  border: 1px solid var(--accent-line); border-radius: 9px; padding: .34rem .52rem; line-height: 1; margin-top: .2rem;
}
.ds-section-title { font-family: var(--font-display); font-size: 1.3rem; font-weight: 600; color: var(--text); letter-spacing: -0.02em; }
.ds-section-lede { font-size: .92rem; color: var(--muted); margin-top: .25rem; max-width: 760px; line-height: 1.55; }
.ds-divider { height: 1px; margin: 2rem 0 0; background: linear-gradient(90deg, var(--border-strong), transparent); }

.ds-chart-head { margin-bottom: .35rem; }
.ds-chart-title { font-family: var(--font-display); font-size: 1rem; font-weight: 600; color: var(--text); }
.ds-chart-caption { font-size: .82rem; color: var(--muted); margin-top: .2rem; line-height: 1.5; }

.ds-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px; margin: .2rem 0 1.4rem; }
.ds-kpi {
  position: relative; overflow: hidden; background: var(--surface); border: 1px solid var(--border);
  border-radius: 18px; padding: 1.1rem 1.2rem 1.15rem; box-shadow: var(--shadow-card);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.ds-kpi::after {
  content: ""; position: absolute; left: 0; top: 0; height: 2px; width: 100%;
  background: linear-gradient(90deg, var(--accent), transparent 75%); opacity: 0; transition: opacity .3s ease;
}
.ds-kpi:hover { transform: translateY(-3px); border-color: var(--accent-line); box-shadow: var(--shadow-lift); }
.ds-kpi:hover::after { opacity: 1; }
.ds-kpi-value {
  font-family: var(--font-display); font-size: 1.85rem; font-weight: 600; color: var(--text);
  letter-spacing: -0.02em; font-variant-numeric: tabular-nums; margin: .45rem 0 .2rem; line-height: 1.1;
}
.ds-kpi.sm .ds-kpi-value { font-size: 1.45rem; }
.ds-kpi-sub { font-size: .8rem; color: var(--muted); line-height: 1.4; }

.ds-meter { height: 6px; border-radius: 999px; background: var(--surface-2); overflow: hidden; margin-top: .8rem; }
.ds-meter > span { display: block; height: 100%; border-radius: 999px; background: var(--accent); transition: width .9s cubic-bezier(.2,.7,.2,1); }

.ds-model {
  height: 100%; border-radius: 20px; border: 1px solid var(--accent-line); padding: 1.4rem 1.5rem;
  background: linear-gradient(160deg, var(--accent-soft), transparent 55%), var(--surface); box-shadow: var(--shadow-card);
}
.ds-model-name { font-family: var(--font-display); font-size: 1.75rem; font-weight: 600; color: var(--text); letter-spacing: -0.02em; margin: .35rem 0 .35rem; }
.ds-model-note { font-size: .84rem; color: var(--muted); line-height: 1.5; margin-bottom: .6rem; }
.ds-meter-row { display: grid; grid-template-columns: 96px 1fr 58px; align-items: center; gap: .75rem; margin: .62rem 0; font-size: .86rem; color: var(--text-2); }
.ds-meter-row .val { text-align: right; font-family: var(--font-mono); font-size: .84rem; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
.ds-meter-row .ds-meter { margin-top: 0; }

.ds-table-wrap {
  overflow-x: auto; border: 1px solid var(--border); border-radius: 16px; background: var(--surface);
  box-shadow: var(--shadow-card); margin: .3rem 0 1rem;
}
.ds-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .9rem; color: var(--text); }
.ds-table th {
  text-align: left; font-size: .7rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
  color: var(--muted); background: var(--surface-2); padding: .85rem 1.1rem; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.ds-table td { padding: .92rem 1.1rem; border-bottom: 1px solid var(--border); vertical-align: middle; line-height: 1.45; color: var(--text-2); }
.ds-table td:first-child { color: var(--text); font-weight: 500; }
.ds-table tbody tr:last-child td { border-bottom: none; }
.ds-table tbody tr:nth-child(even) td { background: var(--row-alt); }
.ds-table tbody tr { transition: background .18s ease; }
.ds-table tbody tr:hover td { background: var(--accent-soft); }
.ds-table tr.hl td { background: var(--accent-soft); color: var(--text); }
.ds-table tr.hl td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
.ds-table .num, .ds-table th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ds-table .mono { font-family: var(--font-mono); font-size: .84rem; }
.ds-table .wrap { white-space: normal; word-break: break-word; min-width: 200px; }
.ds-table .best { color: var(--accent); font-weight: 600; }
.ds-best-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); margin-left: .45rem; vertical-align: middle; }

.ds-flow { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: .3rem 0 1.2rem; }
.ds-step {
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1rem .95rem;
  box-shadow: var(--shadow-card); transition: border-color .25s ease, transform .25s ease;
}
.ds-step:hover { border-color: var(--accent-line); transform: translateY(-2px); }
.ds-step-num { font-family: var(--font-mono); font-size: .72rem; color: var(--accent); }
.ds-step-title { font-family: var(--font-display); font-weight: 600; color: var(--text); margin: .35rem 0 .3rem; font-size: .95rem; }
.ds-step-body { font-size: .8rem; color: var(--muted); line-height: 1.5; }

.ds-note {
  display: flex; gap: .75rem; align-items: flex-start; background: var(--surface-2); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); border-radius: 14px; padding: .9rem 1.05rem; color: var(--text-2);
  font-size: .88rem; line-height: 1.6; margin: .5rem 0 1rem;
}
.ds-note strong { color: var(--text); font-weight: 600; }
.ds-pill {
  display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .72rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; border: 1px solid var(--border); background: var(--surface-2); color: var(--text-2);
}
.ds-pill.ok { color: var(--ok); background: var(--ok-soft); border-color: var(--ok); }
.ds-pill.dot::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.ds-empty {
  text-align: center; padding: 2.4rem 1.4rem; border: 1px dashed var(--border-strong); border-radius: 18px;
  background: var(--surface); color: var(--muted); margin: .4rem 0 1rem;
}
.ds-empty-title { font-family: var(--font-display); color: var(--text); font-weight: 600; font-size: 1.05rem; margin-bottom: .35rem; }

.ds-brand { display: flex; gap: .75rem; align-items: center; padding: .2rem .15rem 1.15rem; border-bottom: 1px solid var(--border); margin-bottom: 1.2rem; }
.ds-brand-name { font-family: var(--font-display); font-weight: 600; color: var(--text); font-size: 1rem; letter-spacing: -0.01em; }
.ds-brand-sub { font-size: .74rem; color: var(--muted); margin-top: 2px; }
.ds-side-label { font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: .4rem 0 .4rem; }
.ds-side-foot { margin-top: 1.3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: .78rem; color: var(--muted); line-height: 1.7; }
.ds-side-foot b { color: var(--text-2); font-weight: 600; }

[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }
[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600; }
.ds-verdict { font-family: var(--font-display); font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em; margin: .3rem 0 .2rem; }
.ds-verdict.benign { color: var(--benign); }
.ds-verdict.malignant { color: var(--malignant); }

@keyframes ds-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.ds-rise { animation: ds-rise .55s cubic-bezier(.2,.7,.2,1) both; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

@media (max-width: 1100px) {
  .ds-flow { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 900px) {
  .block-container { padding: 1.3rem 1rem 3rem; }
  .ds-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ds-hero-art { display: none; }
  .ds-hero { padding: 1.6rem; }
}
@media (max-width: 560px) {
  .ds-kpis { grid-template-columns: 1fr; }
  .ds-flow { grid-template-columns: 1fr; }
  .ds-title { font-size: 1.7rem; }
  .ds-page-title { font-size: 1.6rem; }
  .ds-meter-row { grid-template-columns: 78px 1fr 50px; }
}
"""


_FONT_IMPORT = (
    '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@400;500&family=Sora:wght@500;600;700&display=swap");'
)


def css(theme: str) -> str:
    """Complete stylesheet for the active theme. @import must come first."""
    return _FONT_IMPORT + "\n" + _root_block(palette(theme)) + "\n" + _STATIC_CSS


# --------------------------------------------------------------------------- #
# HTML helpers. Every string is emitted without indentation and without blank
# lines, so Streamlit's Markdown renderer treats it as a single HTML block.
# --------------------------------------------------------------------------- #
def _clean(s: str) -> str:
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def esc(value) -> str:
    return _html.escape(str(value))


def hero(kicker: str, title: str, lede: str, chips=()) -> str:
    chip_html = "".join(f'<span class="ds-chip accent">{esc(c)}</span>' for c in chips)
    art = (
        '<svg class="ds-hero-art" viewBox="0 0 280 170" aria-hidden="true">'
        '<g fill="none" stroke="currentColor" stroke-opacity=".16">'
        '<circle cx="210" cy="90" r="30"/><circle cx="210" cy="90" r="60"/>'
        '<circle cx="210" cy="90" r="90"/><path d="M0 90H280M210 0V170"/></g>'
        '<path d="M0 150 C 66 150, 110 112, 150 72 S 230 16, 280 12 L280 170 L0 170Z" '
        'fill="var(--accent)" fill-opacity=".08"/>'
        '<path d="M0 150 C 66 150, 110 112, 150 72 S 230 16, 280 12" fill="none" '
        'stroke="var(--accent)" stroke-width="3" stroke-linecap="round"/></svg>'
    )
    return _clean(f"""
    <div class="ds-hero ds-rise">
    {art}
    <div class="ds-eyebrow ds-eyebrow-accent">{esc(kicker)}</div>
    <div class="ds-title">{esc(title)}</div>
    <p class="ds-lede">{esc(lede)}</p>
    <div class="ds-chip-row">{chip_html}</div>
    </div>
    """)


def page_intro(index: int, title: str, lede: str) -> str:
    return _clean(f"""
    <div class="ds-page-intro ds-rise">
    <div class="ds-eyebrow">Section {index:02d}</div>
    <div class="ds-page-title">{esc(title)}</div>
    <p class="ds-lede">{esc(lede)}</p>
    </div>
    """)


def section(index: str, title: str, lede: str = "") -> str:
    return _clean(f"""
    <div class="ds-section">
    <span class="ds-section-index">{esc(index)}</span>
    <div>
    <div class="ds-section-title">{esc(title)}</div>
    <div class="ds-section-lede">{esc(lede)}</div>
    </div>
    </div>
    """)


def chart_heading(title: str, caption: str = "") -> str:
    return _clean(f"""
    <div class="ds-chart-head">
    <div class="ds-chart-title">{esc(title)}</div>
    <div class="ds-chart-caption">{esc(caption)}</div>
    </div>
    """)


def _meter(value: float) -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    return f'<div class="ds-meter"><span style="width:{pct:.1f}%"></span></div>'


def kpi_row(items: list[dict]) -> str:
    cells = []
    for it in items:
        meter = _meter(it["meter"]) if it.get("meter") is not None else ""
        cells.append(
            '<div class="ds-kpi">'
            f'<div class="ds-eyebrow">{esc(it["label"])}</div>'
            f'<div class="ds-kpi-value">{esc(it["value"])}</div>'
            f'<div class="ds-kpi-sub">{esc(it.get("sub", ""))}</div>'
            f"{meter}</div>"
        )
    return _clean('<div class="ds-kpis">' + "".join(cells) + "</div>")


def metric_tile(label: str, value: str, sub: str = "", meter: float | None = None) -> str:
    bar = _meter(meter) if meter is not None else ""
    return _clean(
        '<div class="ds-kpi sm">'
        f'<div class="ds-eyebrow">{esc(label)}</div>'
        f'<div class="ds-kpi-value">{esc(value)}</div>'
        f'<div class="ds-kpi-sub">{esc(sub)}</div>{bar}</div>'
    )


def model_card(name: str, note: str, metrics: dict, params: dict) -> str:
    rows = []
    for label, value in metrics.items():
        rows.append(
            '<div class="ds-meter-row">'
            f"<span>{esc(label)}</span>"
            f'{_meter(value)}'
            f'<span class="val">{value:.4f}</span>'
            "</div>"
        )
    chips = "".join(
        f'<span class="ds-chip mono">{esc(k)} = {esc(v)}</span>' for k, v in params.items()
    )
    return _clean(f"""
    <div class="ds-model ds-rise">
    <div class="ds-eyebrow ds-eyebrow-accent">Selected model</div>
    <div class="ds-model-name">{esc(name)}</div>
    <div class="ds-model-note">{esc(note)}</div>
    {''.join(rows)}
    <div class="ds-eyebrow" style="margin-top:1.1rem">Tuned hyperparameters</div>
    <div class="ds-chip-row" style="margin-top:.6rem">{chips}</div>
    </div>
    """)


def flow(steps: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="ds-step"><div class="ds-step-num">{esc(num)}</div>'
        f'<div class="ds-step-title">{esc(title)}</div>'
        f'<div class="ds-step-body">{esc(body)}</div></div>'
        for num, title, body in steps
    )
    return _clean(f'<div class="ds-flow">{cells}</div>')


def note(text_html: str) -> str:
    """text_html may contain <strong>; the caller is responsible for escaping user data."""
    return _clean(f'<div class="ds-note"><div>{text_html}</div></div>')


def empty_state(title: str, text: str) -> str:
    return _clean(
        f'<div class="ds-empty"><div class="ds-empty-title">{esc(title)}</div><div>{esc(text)}</div></div>'
    )


def pill(text: str, ok: bool = True) -> str:
    cls = "ds-pill dot ok" if ok else "ds-pill dot"
    return f'<span class="{cls}">{esc(text)}</span>'


def _default_format(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table_html(
    df: pd.DataFrame,
    *,
    index_label: str = "",
    formats: dict | None = None,
    best: dict | None = None,
    highlight_row=None,
    mono: tuple | list = (),
    wrap: tuple | list = (),
    right: tuple | list = (),
) -> str:
    """
    Render a DataFrame as a designed table.

    formats: column -> callable (value -> str). Alignment follows the original dtype.
    best: column -> "max" | "min"; the best value gets an accent marker.
    highlight_row: index label to highlight, e.g. the selected model.
    """
    formats = formats or {}
    best = best or {}
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])} | set(right)

    best_idx = {}
    for col, how in best.items():
        series = df[col]
        best_idx[col] = series.idxmax() if how == "max" else series.idxmin()

    header = "".join(
        f'<th class="{"num" if c in numeric else ""}">{esc(c)}</th>' for c in df.columns
    )
    body = []
    for idx, row in df.iterrows():
        row_cls = ' class="hl"' if highlight_row is not None and idx == highlight_row else ""
        cells = [f'<td>{esc(idx)}</td>']
        for col in df.columns:
            raw = row[col]
            text = formats[col](raw) if col in formats else _default_format(raw)
            classes = []
            if col in numeric:
                classes.append("num")
            if col in mono:
                classes.append("mono")
            if col in wrap:
                classes.append("wrap")
            marker = ""
            if col in best_idx and idx == best_idx[col]:
                classes.append("best")
                marker = '<span class="ds-best-dot"></span>'
            cls = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{cls}>{esc(text)}{marker}</td>")
        body.append(f"<tr{row_cls}>" + "".join(cells) + "</tr>")

    return _clean(
        '<div class="ds-table-wrap"><table class="ds-table"><thead><tr>'
        f"<th>{esc(index_label)}</th>{header}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def sidebar_brand() -> str:
    mark = (
        '<svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">'
        '<rect width="36" height="36" rx="11" style="fill:var(--accent)"/>'
        '<path d="M9 27 C 14 14, 21 10, 28 10" fill="none" stroke="#FFFFFF" '
        'stroke-width="2.4" stroke-linecap="round"/>'
        '<circle cx="28" cy="10" r="2.4" style="fill:#FFFFFF"/></svg>'
    )
    return _clean(f"""
    <div class="ds-brand">
    {mark}
    <div>
    <div class="ds-brand-name">Tumour Classifier</div>
    <div class="ds-brand-sub">Breast Cancer Wisconsin · Original</div>
    </div>
    </div>
    """)


def sidebar_footer(selected: str, test_rows: int, clean_rows: int) -> str:
    return _clean(f"""
    <div class="ds-side-foot">
    <div><b>Selected model</b><br>{esc(selected)}</div>
    <div style="margin-top:.6rem"><b>Clean samples</b> {clean_rows:,} · <b>Test set</b> {test_rows:,}</div>
    <div style="margin-top:.6rem">Educational analysis of a published dataset. Not a diagnostic tool.</div>
    </div>
    """)
