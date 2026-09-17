# Alternative defense app

This standalone alternative to `streamlit_app/` is a 13-part defense presentation followed by a working X-ray classifier. Its evidence pages use one dominant visual at a time and size graphics against the browser viewport.

Run it from the project root:

```bash
streamlit run streamlit_app_v2/app.py
```

The presentation reads current model performance from JSON artifacts under `reports/`. The final page loads compatible saved models from `models/` and produces a prediction, class probabilities, and a Grad-CAM overlay.

