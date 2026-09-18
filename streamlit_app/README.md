# Streamlit defense presentation

Run the app from the repository root:

```bash
.venv/bin/streamlit run streamlit_app/streamlit_app.py
```

The app presents the project in seven short sections followed by a live image-classification demo. Metrics are read from the saved JSON evaluation artifacts at render time. The demo compares a Simple CNN, EfficientNetB0, fine-tuned EfficientNetB0, and two modes for the materialized EfficientNetB0 across one or more uploaded images, with a Grad-CAM overlay for every prediction. Its scan-only mode accepts the X-ray directly; its scan-and-mask mode applies the exact lungs-only preprocessing used during training.

See `PAGES.md` for the suggested 20-minute run of show.

## Docker / Dokploy

Build and run locally:

```bash
docker compose -f streamlit_app/docker-compose.yml up --build
```

Open `http://localhost:8501`.

For Dokploy, create a Compose application and point it at `streamlit_app/docker-compose.yml` in this repository. The build context is the repository root, so the image includes `src/`, the four demo models, report metrics, figures, and the PDF. Raw dataset files and training checkpoints are excluded via `.dockerignore`.

Recommended server resources: at least 2 GB RAM for TensorFlow model loading and inference.
