from pathlib import Path

import streamlit as st

CLASS_COLORS = {
    "COVID": "#d1495b",
    "Lung_Opacity": "#edae49",
    "Normal": "#00798c",
    "Viral Pneumonia": "#8e7dbe",
}

BASE_CSS = """
<style>
header[data-testid="stHeader"] {
    background: transparent;
}
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 1.1rem;
    padding-bottom: 1.1rem;
    padding-left: clamp(1.2rem, 5vw, 5rem);
    padding-right: clamp(1.2rem, 5vw, 5rem);
    max-width: 100% !important;
    min-height: 92vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    justify-content: safe center;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}
div[data-testid="stHorizontalBlock"] {
    gap: 1.1rem;
}
hr {
    margin: 0.4rem 0 !important;
}
div[data-testid="stImage"] img {
    max-height: 30vh;
    max-width: 100%;
    width: auto;
    height: auto;
    display: block;
    margin: 0 auto;
    object-fit: contain;
}
div[data-testid="stTabs"] {
    margin-top: 0.1rem;
}
div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 0.6rem;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}
h1 {
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
    padding-bottom: 0;
}
h2 {
    font-size: 1.5rem;
    margin-top: 0.1rem;
    margin-bottom: 0.3rem;
    padding-bottom: 0;
}
h3 {
    font-size: 1.25rem;
    margin-top: 0.05rem;
    margin-bottom: 0.2rem;
}
p, li {
    font-size: 1.05rem;
    line-height: 1.45;
}
[data-testid="stCaptionContainer"] {
    font-size: 0.98rem;
}
[data-testid="stMetricValue"] {
    font-size: 2.1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.92rem;
}
[data-testid="stMetricDelta"] {
    font-size: 0.85rem;
}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p,
section[data-testid="stSidebar"] span {
    font-size: 1.02rem;
}
.disclaimer-banner {
    background: #fff4e5;
    border: 1px solid #f0b429;
    border-left: 6px solid #f0b429;
    border-radius: 8px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.95rem;
    line-height: 1.4;
}
.key-finding {
    background: #eef2ff;
    border-left: 6px solid #4c51bf;
    border-radius: 8px;
    padding: 0.7rem 1.1rem;
    margin: 0.3rem 0 0.5rem 0;
    font-size: 1.0rem;
    line-height: 1.4;
}
.step-card {
    border: 1px solid rgba(120,120,120,0.28);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    height: 100%;
    min-height: 148px;
}
.step-card .step-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.6;
    margin-bottom: 0.15rem;
}
.step-card .step-title {
    font-weight: 700;
    font-size: 1.12rem;
    margin-bottom: 0.3rem;
    line-height: 1.25;
}
.step-card .step-body {
    font-size: 0.92rem;
    opacity: 0.85;
    line-height: 1.35;
}
.status-pill {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.08rem 0.5rem;
    border-radius: 999px;
    margin-bottom: 0.35rem;
}
.status-done {
    background: #e3f6e8;
    color: #1a7f37;
}
.status-partial {
    background: #fff4e5;
    color: #9a6700;
}
.cover-hero {
    text-align: center;
    padding: 3rem 2rem 2.6rem 2rem;
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(76,81,191,0.07), rgba(76,81,191,0.0));
    margin-bottom: 1rem;
}
.cover-eyebrow {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    opacity: 0.6;
    margin-bottom: 0.9rem;
}
.cover-title {
    font-size: 3.4rem;
    font-weight: 800;
    margin: 0 0 1rem 0;
    line-height: 1.15;
}
.cover-authors {
    font-size: 1.5rem;
    font-weight: 600;
    color: #4c51bf;
    margin-bottom: 1.1rem;
}
.cover-context {
    font-size: 1.15rem;
    line-height: 1.55;
    max-width: 900px;
    margin: 0 auto;
    opacity: 0.9;
}
.narrative-card {
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    height: 100%;
}
.narrative-card .narrative-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.narrative-card .narrative-body {
    font-size: 0.88rem;
    line-height: 1.3;
}
.narrative-challenge {
    background: #fdf2f2;
    border-left: 6px solid #d1495b;
}
.narrative-challenge .narrative-label {
    color: #d1495b;
}
.narrative-did {
    background: #eef2ff;
    border-left: 6px solid #4c51bf;
}
.narrative-did .narrative-label {
    color: #4c51bf;
}
.narrative-found {
    background: #e3f6e8;
    border-left: 6px solid #1a7f37;
}
.narrative-found .narrative-label {
    color: #1a7f37;
}
.mini-step {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    font-size: 0.98rem;
    padding: 0.3rem 0;
}
.mini-step .mini-step-num {
    font-weight: 700;
    color: #4c51bf;
    min-width: 1.4rem;
}
.app-footer {
    margin-top: 0.8rem;
    padding-top: 0.6rem;
    border-top: 1px solid rgba(120,120,120,0.25);
    font-size: 0.8rem;
    opacity: 0.6;
}
section[data-testid="stSidebar"] {
    min-width: 240px;
}
.figure-missing {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 160px;
    border: 1px dashed rgba(120,120,120,0.4);
    border-radius: 10px;
    color: rgba(50,50,50,0.55);
    font-size: 0.85rem;
    line-height: 1.4;
    text-align: center;
    padding: 1rem;
}
@media (max-width: 640px) {
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
        min-height: 0;
    }
    .cover-title {
        font-size: 2.1rem;
    }
    .cover-authors {
        font-size: 1.15rem;
    }
    .cover-context {
        font-size: 1rem;
    }
    .step-card {
        min-height: 0;
    }
}
@media (max-width: 380px) {
    .cover-hero {
        padding: 2rem 1.2rem 1.8rem 1.2rem;
    }
    .cover-title {
        font-size: 1.6rem;
    }
    .cover-authors {
        font-size: 1rem;
    }
    .cover-context {
        font-size: 0.92rem;
    }
    h1 {
        font-size: 1.7rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    .step-card {
        padding: 0.7rem 0.85rem;
    }
    .figure-missing {
        min-height: 110px;
        font-size: 0.78rem;
    }
}
</style>
"""


def configure_page(title: str, icon: str = None) -> None:
    st.set_page_config(page_title=f"{title} | COVID-19 CXR Defense", page_icon=icon, layout="wide")
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def disclaimer_banner() -> None:
    st.markdown(
        '<div class="disclaimer-banner">'
        "Educational decision-support study, not a clinically validated diagnostic system. "
        "Nothing here is medical advice."
        "</div>",
        unsafe_allow_html=True,
    )


def key_finding(text: str) -> None:
    st.markdown(f'<div class="key-finding">{text}</div>', unsafe_allow_html=True)


def cover_hero(eyebrow: str, title: str, authors: str, context: str) -> None:
    st.markdown(
        f'<div class="cover-hero">'
        f'<div class="cover-eyebrow">{eyebrow}</div>'
        f'<div class="cover-title">{title}</div>'
        f'<div class="cover-authors">{authors}</div>'
        f'<div class="cover-context">{context}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def step_card(label: str, title: str, body: str, status: str = "done") -> None:
    status_class = "status-done" if status == "done" else "status-partial"
    status_text = "Done" if status == "done" else "Partial"
    st.markdown(
        f'<div class="step-card">'
        f'<div class="status-pill {status_class}">{status_text}</div>'
        f'<div class="step-label">{label}</div>'
        f'<div class="step-title">{title}</div>'
        f'<div class="step-body">{body}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def narrative_row(challenge: str, did: str, found: str) -> None:
    cols = st.columns(3, gap="small")
    items = [
        ("challenge", "Challenge", challenge),
        ("did", "What we did", did),
        ("found", "What we found", found),
    ]
    for col, (key, label, text) in zip(cols, items):
        col.markdown(
            f'<div class="narrative-card narrative-{key}">'
            f'<div class="narrative-label">{label}</div>'
            f'<div class="narrative-body">{text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def mini_steps(steps) -> None:
    cols = st.columns(4, gap="small")
    for index, label in enumerate(steps):
        col = cols[index % 4]
        col.markdown(
            f'<div class="mini-step"><span class="mini-step-num">{index + 1}</span>{label}</div>',
            unsafe_allow_html=True,
        )


def safe_image(path, caption: str = None, **kwargs) -> None:
    resolved = Path(path)
    if not resolved.exists():
        label = caption or f"Figure not available: {resolved.name}"
        st.markdown(f'<div class="figure-missing">{label}</div>', unsafe_allow_html=True)
        return
    st.image(str(resolved), caption=caption, **kwargs)


def app_footer() -> None:
    st.markdown(
        '<div class="app-footer">COVID-19 Radiography Database (Kaggle, tawsifurrahman) '
        "&middot; arXiv:2003.13865 &middot; doi:10.1016/j.compbiomed.2021.105002</div>",
        unsafe_allow_html=True,
    )
