# Presentation Structure

Defense presentation for the COVID-19 chest X-ray classification project, by Berfin and Mert.
Each content page follows the same narrative: the challenge we faced, what we did about it,
and what we found or experienced. Built one page at a time; status is tracked below.

## Page list

| # | File | Topic | Status |
| --- | --- | --- | --- |
| - | `Home.py` | Cover slide: title, authors, project context, numbers, steps | Done |
| 1 | `pages/1_Dataset_and_EDA.py` | Dataset & EDA | Done |
| 2 | `pages/2_Preprocessing.py` | Preprocessing | Done |
| 3 | `pages/3_Classical_and_Baseline_Models.py` | Classical & Baseline Models | Done |
| 4 | `pages/4_Deep_CNN_from_Scratch.py` | Deep CNN From Scratch | Done |
| 5 | `pages/5_Transfer_Learning.py` | Transfer Learning | Done |
| 6 | `pages/6_Interpretability_and_Limitations.py` | Interpretability & Limitations | Done |
| 7 | `pages/7_Conclusion.py` | Conclusion | Done |
| 8 | `pages/8_Live_Demo.py` | Live Demo | Done |

## Page details

### Home — Cover

Title, "Berfin & Mert" as authors, one-line project context, non-clinical disclaimer,
a row of headline numbers (image count, class count, models trained, best accuracy),
and a plain numbered list of the 8 pipeline steps.

### 1. Dataset & EDA

- **Challenge:** the dataset is stitched together from different source repositories per
  class (RSNA, Kaggle Pneumonia, SIRM/GitHub/Twitter for COVID), a known recipe for hidden
  shortcuts.
- **What we did:** audited structural integrity, class balance (chi-square test), RGB-encoding
  check, duplicate detection (pHash), brightness/contrast tests (Kruskal-Wallis, Cohen's d).
- **What we found:** 140 stray RGB images only in Viral Pneumonia, 59 duplicates concentrated
  in COVID, confirmed class imbalance, and early statistical hints of the shortcut-learning
  risk that later gets proven causally in Step 6.

### 2. Preprocessing

- **Challenge:** fix what EDA exposed without introducing new bias, and make it fully
  reproducible.
- **What we did:** deduplicate before splitting, stratified 70/15/15 split (seed 42), grayscale
  standardization, nearest-neighbor mask alignment, per-image normalization, mask-safe
  augmentation pipeline.
- **What we experienced:** built 78 automated tests, including a bit-identical full-pipeline
  rebuild check; reproducibility treated as a first-class requirement, not an afterthought.

### 3. Classical & Baseline Models

- **Challenge:** before touching deep learning, what is the honest floor? What can trivial
  models already do?
- **What we did:** dummy classifier, logistic regression, histogram gradient boosting on
  downsampled raw pixels; a region-confound protocol (lungs-only vs. background-only crops).
- **What we experienced:** a linear model hit 71% accuracy, and gradient boosting on raw
  pixels hit 86%, higher than a naive CNN. A humbling signal that a lot of separability exists
  even without deep learning, some of it likely from non-lung regions.

### 4. Deep CNN From Scratch

- **Challenge:** with no pretrained weights available, how much do architecture capacity
  and image region actually matter for a CNN trained from scratch?
- **What we did:** trained LeNet-5 (full image) and a lung-centred ROI variant of the same
  simple 3-block CNN used as the full-image baseline, same split, seed, and class weighting.
- **What we experienced:** LeNet-5 lands within a point of the full-image simple CNN despite
  a very different, position-sensitive design; cropping to a lung-centred ROI costs 3.3
  accuracy points and drops COVID recall by nearly 13 points, an early hint of the background
  shortcut confirmed causally in Step 6.

### 5. Transfer Learning

- **Challenge:** does ImageNet-pretrained knowledge transfer to grayscale medical X-rays, a
  very different visual domain?
- **What we did:** EfficientNetB0 frozen backbone, then augmentation and class-weighting
  ablations, then end-to-end fine-tuning.
- **What we experienced:** frozen backbone alone hit 89.7%; horizontal flip hurt performance
  (X-rays are not meaningfully flip-invariant); fine-tuning reached 92.36%, strong but still
  short of the from-scratch CNN.

### 6. Interpretability & Limitations

The trust investigation, likely the strongest section of the defense.

- **Challenge:** high accuracy is not evidence of trustworthiness. Is the model reading lungs
  or reading the dataset?
- **What we did:** Grad-CAM plus a lung-focus-vs-chance metric, then a causal test that retrains
  with the background forcibly zeroed out, plus a shape-only (mask silhouette) ablation.
- **What we experienced:** the key finding. Masking the background collapsed COVID recall from
  90.3% to 57.0% while other classes barely moved. Causal proof of dataset-specific shortcut
  learning, not just a correlational Grad-CAM hunch.

### 7. Conclusion

Scientific takeaways, practical and deployment framing (what external validation would be
required before any of this could be trusted), limitations, and what we would do differently.

### 8. Live Demo

Upload an X-ray, pick a saved model, see the prediction and its Grad-CAM overlay. Hands-on
closing for the defense.

## Design conventions

- Each page fills the browser viewport like a slide: full width, vertically centered content,
  Streamlit's default chrome hidden.
- Shared building blocks live in `lib/`: `paths.py` (project paths), `metrics.py` (load and
  summarize metrics JSON), `ui.py` (page config, CSS, cards, banners, `narrative_row` for the
  Challenge / What we did / What we found pattern).
- No code comments, no emojis, mobile-responsive layout.
- Numbers are sourced from the metrics JSON files under `reports/`, not retyped from the LaTeX
  report, since the LaTeX text is behind the current state of trained models in places (for
  example, the from-scratch CNN in Step 4 outperforms the fine-tuned transfer model described
  as the project's best result in `reports/latex/chapters/transfer_learning.tex`).
