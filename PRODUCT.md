# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: bootcamp jury/instructors evaluating the final-project defense, in a hybrid situation — a presenter-led walkthrough on a shared screen/projector during the live defense, and self-paced review by each juror on their own device before or after it. The app must stand on its own without a narrator filling gaps.

Secondary: recruiters/employers reviewing the app afterward as a portfolio piece.

Maintainers/presenters: the two authors, Berfin and Mert.

## Product Purpose

The underlying project trains and evaluates image classification models that distinguish COVID-19 chest X-rays from Normal, Viral Pneumonia, and Lung Opacity cases (COVID-19 Radiography Database, Kaggle). `streamlit_app/` is the defense-presentation surface: a slide-by-slide walkthrough of the full pipeline (Dataset & EDA, Preprocessing, Classical & Baseline Models, Deep CNN From Scratch, Transfer Learning, Interpretability & Limitations, Conclusion, Live Demo), success meaning a viewer understands both what was built and how trustworthy it actually is.

## Positioning

Not a commercial product. What a neighboring bootcamp defense could not truthfully copy: a causal (not just correlational) proof of dataset-specific shortcut learning — masking the background collapsed COVID recall from 90.3% to 57.0% — surfaced as a first-class finding rather than buried as a caveat.

## Operating Context

Live bootcamp defense presentation, then a self-paced artifact, then a portfolio piece. Every page follows the same three-part narrative (Challenge / What we did / What we found). Numbers displayed in the app are sourced live from the metrics JSON files under `reports/` (baseline, cnn, transfer_learning), not retyped from the LaTeX report text, because the LaTeX report is already behind the current best-trained models in places (e.g. the from-scratch CNN in Step 4 now outperforms the fine-tuned transfer model the LaTeX text still calls the project's best result).

## Capabilities and Constraints

- 8 pages total: `Home.py` (cover) plus 7 numbered pages under `pages/`, tracked in `streamlit_app/PAGES.md`. 2 done (Home, Dataset & EDA), 6 pending.
- Each page fills the viewport like a slide: full width, vertically centered, Streamlit chrome (menu, footer, header background) hidden.
- Shared building blocks live in `streamlit_app/lib/`: `paths.py` (project paths), `metrics.py` (load/summarize metrics JSON), `ui.py` (page config, CSS, `cover_hero`, `step_card`, `narrative_row`, `mini_steps`, banners, footer). New pages reuse these rather than inventing bespoke markup.
- No code comments, no emojis (standing rule for this codebase).
- Layout must stay responsive on mobile (existing `@media (max-width: 640px)` rules in `lib/ui.py` are the baseline to extend, not replace).
- Colorblind-safe palette is explicitly out of scope — not a requirement for this tool.
- The non-clinical disclaimer banner must stay visible: this is an educational decision-support study, not a clinically validated diagnostic system, and nothing in the app is medical advice.
- Live Demo page (`8_Live_Demo.py`, pending): upload an X-ray, pick a saved model from `models/`, see the prediction and its Grad-CAM overlay. Exact interaction constraints (which saved models are exposed, upload limits) are undecided — treat as open until specified.

## Brand Commitments

Title "COVID-19 Chest X-Ray Classification", byline "Berfin & Mert", framed as a Data Science Bootcamp final-project defense. Existing visual identity already committed in code: indigo primary accent (`#4c51bf`, set in `.streamlit/config.toml` and reused throughout `lib/ui.py`), light theme, class-identity colors (COVID `#d1495b`, Lung_Opacity `#edae49`, Normal `#00798c`, Viral Pneumonia `#8e7dbe`).

## Evidence On Hand

- Figures: `reports/latex/figures/*.png` (class distribution, viral pneumonia file size, duplicate distribution, brightness/contrast box plots and scatter).
- Metrics JSON under `reports/baseline/`, `reports/cnn/`, `reports/transfer_learning/`.
- Write-up: `reports/transfer_learning/README.md` (Grad-CAM / lung-focus / background-masking findings).
- No customer testimonials, case studies, or press — not applicable to this project; do not fabricate any.

## Product Principles

1. Every content page follows Challenge / What we did / What we found — no page breaks that structure.
2. Numbers are read from `reports/` metrics JSON at render time, never retyped or invented, and the app is the source of truth over the LaTeX report when the two disagree.
3. Honest reporting of limitations (shortcut learning, non-clinical status, uncertainty) is treated as central content, not a footnote.
4. The app must work whether or not a presenter is narrating — self-paced and presenter-led are equally first-class.
5. New pages extend the shared `lib/` components already established rather than introducing one-off styling per page.

## Accessibility & Inclusion

No specific accessibility standard is required. Colorblind-safe color choices were explicitly decided as out of scope for this tool.
