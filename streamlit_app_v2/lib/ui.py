from html import escape
from pathlib import Path

import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600;9..144,750&family=Manrope:wght@400;500;600;700&display=swap');
:root {
    --ink: #1d2525;
    --paper: #f3f0e8;
    --paper-deep: #e8e2d6;
    --red: #b43b2f;
    --amber: #d69c32;
    --teal: #2e6862;
    --line: rgba(29, 37, 37, .19);
}
html, body, [class*="css"] { font-family: "Manrope", sans-serif; color: var(--ink); }
html, body, .stApp { height: 100%; }
.stApp {
    background-color: var(--paper);
    background-image: radial-gradient(rgba(29,37,37,.055) .7px, transparent .7px);
    background-size: 7px 7px;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1560px !important;
    padding: clamp(.8rem, 2dvh, 1.7rem) clamp(1.2rem, 4vw, 4.8rem) clamp(.7rem, 1.5dvh, 1.25rem);
    display: flex;
    flex-direction: column;
    justify-content: center;
}
[data-testid="stMain"]:has(.viewport-marker) { overflow: clip; }
[data-testid="stMainBlockContainer"]:has(.viewport-marker) {
    box-sizing: border-box;
    height: 100dvh;
    min-height: 100dvh;
    max-height: 100dvh;
    overflow: clip;
}
[data-testid="stMainBlockContainer"]:has(.viewport-marker) > div,
[data-testid="stMainBlockContainer"]:has(.viewport-marker) [data-testid="stVerticalBlock"] {
    gap: clamp(.25rem, .65dvh, .6rem);
}
.viewport-marker { display: none; }
[data-testid="stMainBlockContainer"]:has(.evidence-page),
[data-testid="stMainBlockContainer"]:has(.app-page) {
    justify-content: flex-start;
    padding-top: clamp(1rem, 2.2dvh, 2rem);
}
section[data-testid="stSidebar"] {
    background: #202828;
    border-right: 1px solid rgba(255,255,255,.12);
}
section[data-testid="stSidebar"] * { color: #ede9de !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] { background: rgba(255,255,255,.18); }
section[data-testid="stSidebar"] a[aria-current="page"] {
    background: #b43b2f !important;
    border-radius: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { border-radius: 0 !important; }
h1, h2, h3 { font-family: "Fraunces", Georgia, serif; letter-spacing: -.035em; }
h1 { font-size: clamp(2.15rem, min(4.2vw, 5.6dvh), 4.7rem); line-height: .98; margin: 0; padding: 0; }
h2 { font-size: clamp(1.65rem, 2.8vw, 2.7rem); }
h3 { font-size: clamp(1.05rem, 1.4vw, 1.4rem); margin: .25rem 0; }
p, li { font-size: clamp(.86rem, 1vw, 1.03rem); line-height: 1.45; }
[data-testid="stCaptionContainer"] { font-family: "DM Mono", monospace; font-size: .78rem; }
.deck-kicker, .mono, .slide-index, .chip, .card-label, .result-label, .source-line {
    font-family: "DM Mono", monospace;
    text-transform: uppercase;
    letter-spacing: .09em;
}
.slide-index { color: var(--red); font-size: .68rem; margin-bottom: clamp(.2rem, .6dvh, .55rem); }
.deck-kicker { font-size: .75rem; margin-bottom: 1rem; }
.deck-title { max-width: 1080px; }
.deck-title .accent { color: var(--red); font-style: italic; }
.deck-subtitle { max-width: 760px; font-size: 1.22rem; line-height: 1.6; margin: 1.4rem 0 1.8rem; }
.byline { display: flex; gap: 1.2rem; align-items: center; font-family: "DM Mono", monospace; font-size: .8rem; }
.byline::before { content: ""; width: 54px; height: 4px; background: var(--red); }
.page-lede { max-width: 1080px; font-size: clamp(.92rem, 1.15vw, 1.16rem); line-height: 1.4; margin: clamp(.35rem, .8dvh, .7rem) 0 clamp(.45rem, 1dvh, .8rem); }
[data-testid="stMainBlockContainer"]:has(.evidence-page) h1 {
    font-size: clamp(2rem, min(3.5vw, 4.8dvh), 3.8rem);
}
[data-testid="stMainBlockContainer"]:has(.evidence-page) .page-lede {
    max-width: 1180px;
    margin-bottom: clamp(.35rem, .75dvh, .7rem);
}
.editorial-rule { border-top: 1px solid var(--ink); margin: 1.1rem 0 1.25rem; }
.trio { display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--line); margin: .1rem 0 clamp(.35rem, .8dvh, .65rem); }
.trio-card { padding: clamp(.45rem, .8dvh, .7rem) .85rem; border-right: 1px solid var(--line); min-height: 76px; }
.trio-card:last-child { border-right: 0; }
.card-label { color: var(--red); font-size: .6rem; margin-bottom: .25rem; }
.card-copy { font-size: clamp(.72rem, .78vw, .84rem); line-height: 1.32; }
.number-grid { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); }
.number-cell { padding: clamp(.6rem, 1.2dvh, 1rem) .8rem clamp(.55rem, 1dvh, .9rem) 0; }
.big-number { font-family: "Fraunces", Georgia, serif; font-size: clamp(1.8rem, min(3.3vw, 4.8dvh), 3.7rem); line-height: 1; color: var(--red); }
.number-label { font-family: "DM Mono", monospace; font-size: .72rem; text-transform: uppercase; margin-top: .45rem; }
.callout { border-left: 6px solid var(--red); background: #ebe4d7; padding: clamp(.5rem, .9dvh, .75rem) .9rem; margin: .35rem 0 .5rem; font-size: clamp(.75rem, .82vw, .9rem); line-height: 1.35; }
.callout strong { font-family: "Fraunces", Georgia, serif; font-size: 1.02rem; }
.thesis { font-family: "Fraunces", Georgia, serif; font-size: clamp(1.8rem, 3.6vw, 3.6rem); line-height: 1.14; max-width: 1100px; margin: .8rem 0 1.6rem; }
.thesis em { color: var(--red); }
.pathway { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: var(--ink); border: 1px solid var(--ink); }
.path-step { background: var(--paper); padding: .9rem; min-height: 130px; }
.path-num { font-family: "Fraunces", Georgia, serif; color: var(--red); font-size: 1.8rem; }
.path-title { font-weight: 700; margin: .25rem 0; }
.path-copy { font-size: .78rem; line-height: 1.35; }
.verdict { background: var(--ink); color: var(--paper); padding: clamp(.8rem, 1.6dvh, 1.4rem) clamp(1rem, 2vw, 2rem); margin: .35rem 0; }
.verdict .result-label { color: #e7ac47; font-size: .72rem; }
.verdict .result { font-family: "Fraunces", Georgia, serif; font-size: clamp(1.55rem, min(2.7vw, 3.8dvh), 3rem); line-height: 1.06; margin-top: .35rem; }
.disclaimer { border: 1px solid var(--red); padding: .45rem .7rem; font-family: "DM Mono", monospace; font-size: .66rem; margin: .35rem 0 .5rem; }
.source-line { border-top: 1px solid var(--line); margin-top: .35rem; padding-top: .3rem; font-size: .56rem; opacity: .65; }
.figure-frame { background: #202828; padding: .7rem; border: 1px solid #111; box-shadow: 8px 8px 0 #d7d0c3; }
div[data-testid="stImage"] { display: flex; flex-direction: column; align-items: center; }
div[data-testid="stImage"] img { width: 100%; max-height: clamp(300px, 52dvh, 620px); object-fit: contain; }
[data-testid="stMainBlockContainer"]:has(.evidence-page) div[data-testid="stImage"] img {
    max-height: clamp(390px, 68dvh, 760px);
}
[data-testid="stMainBlockContainer"]:has(.cover-page) div[data-testid="stImage"] img { max-height: 44dvh; }
[data-testid="stMetric"] { border-top: 3px solid var(--ink); padding: clamp(.35rem, .7dvh, .55rem) .12rem; }
[data-testid="stMetricValue"] { font-family: "Fraunces", Georgia, serif; font-size: clamp(1.45rem, min(2.4vw, 3.4dvh), 2.7rem); color: var(--red); }
[data-testid="stMetricLabel"] { font-family: "DM Mono", monospace; text-transform: uppercase; font-size: .68rem; }
[data-testid="stMetricDelta"] { font-size: .68rem; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); }
div[data-testid="stTabs"] button[data-baseweb="tab"] { height: 2.2rem; padding: .25rem .75rem; }
div[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: clamp(.25rem, .6dvh, .55rem); }
[data-testid="stVegaLiteChart"] { height: clamp(320px, 54dvh, 630px) !important; min-height: 0; }
[data-testid="stMainBlockContainer"]:has(.evidence-page) [data-testid="stVegaLiteChart"] {
    height: clamp(400px, 67dvh, 760px) !important;
}
.stButton > button, .stDownloadButton > button {
    border: 1px solid var(--ink); border-radius: 0; background: var(--ink); color: var(--paper);
    font-family: "DM Mono", monospace; text-transform: uppercase; letter-spacing: .05em;
}
.stButton > button:hover { background: var(--red); color: white; border-color: var(--red); }
[data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] { border-radius: 0 !important; }
[data-testid="stFileUploaderDropzone"] { padding: clamp(.45rem, .8dvh, .7rem) .8rem; min-height: 72px; }
@keyframes reveal { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
.block-container > div { animation: reveal .42s ease-out both; }
@media (max-width: 800px) {
    [data-testid="stMain"]:has(.viewport-marker) { overflow: auto; }
    [data-testid="stMainBlockContainer"]:has(.viewport-marker) { height: auto; min-height: 100dvh; max-height: none; overflow: visible; padding-top: 2.3rem; }
    .trio, .number-grid, .pathway { grid-template-columns: 1fr; }
    .trio-card { border-right: 0; border-bottom: 1px solid var(--line); min-height: auto; }
    .trio-card:last-child { border-bottom: 0; }
    .path-step { min-height: auto; }
}
@media (min-width: 801px) and (max-height: 800px) {
    .block-container, [data-testid="stMainBlockContainer"] { padding-top: .75rem; padding-bottom: .55rem; }
    .page-lede { margin: .2rem 0 .35rem; }
    .trio-card { min-height: 64px; padding: .35rem .65rem; }
    .card-copy { font-size: .7rem; }
    .source-line { display: none; }
    div[data-testid="stImage"] img { max-height: 47dvh; }
    [data-testid="stVegaLiteChart"] { height: 46dvh !important; }
    [data-testid="stMainBlockContainer"]:has(.evidence-page) div[data-testid="stImage"] img { max-height: 64dvh; }
    [data-testid="stMainBlockContainer"]:has(.evidence-page) [data-testid="stVegaLiteChart"] { height: 62dvh !important; }
}
</style>
"""


def configure() -> None:
    st.set_page_config(
        page_title="Reading the Dataset | COVID-19 X-ray Defense",
        page_icon=":material/radiology:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="mono" style="font-size:.67rem;opacity:.7;margin:.5rem 0 1rem">'
        "Berfin + Mert<br>Defense deck / 2026</div>",
        unsafe_allow_html=True,
    )


def fit_page(mode: str = "visual") -> None:
    st.markdown(f'<span class="viewport-marker {escape(mode)}-page"></span>', unsafe_allow_html=True)


def header(index: str, title: str, lede: str, mode: str = "visual") -> None:
    fit_page(mode)
    st.markdown(f'<div class="slide-index">{escape(index)}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="page-lede">{escape(lede)}</div>', unsafe_allow_html=True)


def evidence_header(index: str, title: str, lede: str) -> None:
    header(index, title, lede, mode="evidence")


def trio(challenge: str, response: str, evidence: str) -> None:
    values = (("The challenge", challenge), ("Our move", response), ("What the evidence said", evidence))
    cards = "".join(
        f'<div class="trio-card"><div class="card-label">{escape(label)}</div>'
        f'<div class="card-copy">{escape(copy)}</div></div>'
        for label, copy in values
    )
    st.markdown(f'<div class="trio">{cards}</div>', unsafe_allow_html=True)


def numbers(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="number-cell"><div class="big-number">{escape(value)}</div>'
        f'<div class="number-label">{escape(label)}</div></div>'
        for value, label in items
    )
    st.markdown(f'<div class="number-grid">{cells}</div>', unsafe_allow_html=True)


def callout(title: str, text: str) -> None:
    st.markdown(
        f'<div class="callout"><strong>{escape(title)}</strong><br>{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def verdict(label: str, text: str) -> None:
    st.markdown(
        f'<div class="verdict"><div class="result-label">{escape(label)}</div>'
        f'<div class="result">{escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer">Educational decision-support study. Not clinically validated. '
        "No output in this app is medical advice.</div>",
        unsafe_allow_html=True,
    )


def image(path: Path, caption: str = "", **kwargs) -> None:
    if not Path(path).exists():
        st.warning(f"Figure unavailable: {Path(path).name}")
        return
    st.image(str(path), caption=caption or None, **kwargs)


def source(text: str = "Metrics loaded from saved project artifacts under reports/.") -> None:
    st.markdown(f'<div class="source-line">{escape(text)}</div>', unsafe_allow_html=True)


def probability_chart(frame) -> None:
    st.bar_chart(frame.set_index("Class")[["Probability"]], height=380, color="#b43b2f")


def metric_chart(frame, category: str = "Model", height: int = 520) -> None:
    value_columns = [column for column in frame.columns if column != category]
    long = frame.melt(id_vars=[category], value_vars=value_columns, var_name="Metric", value_name="Score")
    spec = {
        "height": height,
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusEnd": 3},
                "encoding": {
                    "y": {
                        "field": category,
                        "type": "nominal",
                        "sort": None,
                        "title": None,
                        "axis": {"labelLimit": 290, "labelFontSize": 14},
                    },
                    "yOffset": {"field": "Metric"},
                    "x": {
                        "field": "Score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 1]},
                        "axis": {"format": ".0%", "title": None, "gridOpacity": 0.2},
                    },
                    "color": {
                        "field": "Metric",
                        "type": "nominal",
                        "scale": {"range": ["#b43b2f", "#2e6862"]},
                        "legend": {"orient": "top", "title": None},
                    },
                },
            },
            {
                "mark": {
                    "type": "text",
                    "align": "left",
                    "baseline": "middle",
                    "dx": 7,
                    "fontSize": 13,
                    "fontWeight": 600,
                    "color": "#1d2525",
                },
                "encoding": {
                    "y": {"field": category, "type": "nominal", "sort": None},
                    "yOffset": {"field": "Metric"},
                    "x": {"field": "Score", "type": "quantitative", "scale": {"domain": [0, 1]}},
                    "text": {"field": "Score", "type": "quantitative", "format": ".1%"},
                    "detail": {"field": "Metric"},
                },
            },
        ],
        "config": {
            "background": None,
            "view": {"stroke": None},
            "axis": {"labelFont": "Manrope", "titleFont": "Manrope"},
            "legend": {"labelFont": "Manrope", "labelFontSize": 13},
        },
    }
    st.vega_lite_chart(long, spec, width="stretch")
