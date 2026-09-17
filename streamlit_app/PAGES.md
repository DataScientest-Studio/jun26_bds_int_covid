# Defense run of show

| Screen | Target time | Main message |
| --- | ---: | --- |
| Home | 1:00 | We evaluate both performance and whether the model learns for the right reasons. |
| Dataset | 2:00 | The four-class dataset is useful but class, source, and technical artifacts are entangled. |
| Exploration | 3:00 | Encoding, duplicates, and intensity differences became concrete preprocessing decisions. |
| Preprocessing | 2:00 | Deduplication happened before the reproducible 70/15/15 split. |
| Modelling | 2:30 | Complexity increased from transparent baselines to a compact CNN benchmark and transfer learning. |
| Results | 3:00 | Fine-tuned EfficientNetB0 leads the models included in the presentation. |
| Trust & limits | 4:00 | Background masking caused a COVID-specific recall collapse, revealing shortcut learning. |
| Live demo | 2:30 | Show probabilities and Grad-CAM, then repeat the non-clinical limitation. |

Total target: 20 minutes.

For the live demo, keep one known frontal X-ray ready locally. If inference is slow, explain that the first run loads the model and later runs use the cached copy. Do not present confidence as clinical certainty or Grad-CAM as lesion localization.
