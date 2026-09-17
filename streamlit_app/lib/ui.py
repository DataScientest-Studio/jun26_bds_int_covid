from pathlib import Path

import streamlit as st


CLASS_COLORS = {
    "COVID": "#D1495B",
    "Lung Opacity": "#EDAE49",
    "Normal": "#00798C",
    "Viral Pneumonia": "#8E7DBE",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="COVID-19 chest X-ray classification",
        page_icon=":material/radiology:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(
        """
        <style>
        .block-container {max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem;}
        h1 {letter-spacing: -0.035em;}
        h2, h3 {letter-spacing: -0.02em;}
        [data-testid="stMetric"] {min-width: 170px;}
        [data-testid="stImage"] img {border-radius: 12px;}
        .stCaption {color: #64708A;}
        @media (max-width: 640px) {
            .block-container {padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem;}
            h1 {font-size: 2rem !important;}
        }
        </style>
        """
    )


def page_intro(step: str, title: str, subtitle: str) -> None:
    st.caption(step.upper())
    st.title(title)
    st.markdown(f"### {subtitle}")


def narrative(challenge: str, action: str, finding: str) -> None:
    columns = st.columns(3, gap="medium")
    items = [
        (":material/help:", "Challenge", challenge),
        (":material/build:", "What we did", action),
        (":material/search_check:", "What we found", finding),
    ]
    for column, (icon, label, body) in zip(columns, items):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {icon} {label}")
            st.write(body)


def disclaimer() -> None:
    st.warning(
        "Educational decision-support study only. This application is not clinically validated and does not provide medical advice or a diagnosis.",
        icon=":material/health_and_safety:",
    )


def figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Figure not found: {path.name}")


def footer(section: str) -> None:
    st.caption(f"{section} · Data Science Bootcamp final project · Berfin & Mert")

