FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[app]" pillow

COPY streamlit_app ./streamlit_app
COPY reports ./reports
COPY models/cnn_scratch.keras \
     models/cnn_simple.keras \
     models/transfer_efficientnetb0.keras \
     models/transfer_efficientnetb0_finetuned.keras \
     ./models/

RUN mkdir -p /app/data/arrays

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD [
    "streamlit", "run", "streamlit_app/Home.py",
    "--server.headless", "true",
    "--server.address", "0.0.0.0",
    "--server.port", "8501",
    "--browser.gatherUsageStats", "false"
]
