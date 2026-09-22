# Defense run of show

## 18-minute target

| Screen | Speaker | Target | Explain | Point at |
| --- | --- | ---: | --- | --- |
| Home | Berfin | 0:45 | Performance and trust are separate questions | Central research question and four badges |
| Data audit | Berfin | 3:00 | Four classes, source entanglement, three finding→action decisions | Four X-rays, source table, RGB and duplicate charts |
| Preprocessing | Mert | 1:30 | The audit decisions became a reproducible pipeline | Pipeline badges and paired-transform figure |
| Modelling | Mert | 2:15 | Eight experiments had distinct purposes | Experiment map; fine-tuned and lungs-only rows |
| Results | Mert | 2:15 | Fine-tuned EfficientNetB0 is the retained raw-score winner | Three metrics, winner confusion matrix, transition box |
| Trust & limits | Berfin | 4:00 | Multiple constraints were tried; matched masking exposed a COVID-specific shortcut | Reasoning chain and 90.3% → 57.0% metrics |
| Conclusion | Berfin | 1:30 | Strong classifier, source-confounded evidence, careful future work | Three conclusion cards and future-work list |
| Live demo | Mert | 2:30 | One cached model, four probabilities, honest Grad-CAM caveat | Prediction, probability bars, heatmap |

Target total: 17 minutes 45 seconds, leaving about 2 minutes for interruption or questions.

## Exact handoffs

- Berfin → Mert, after Data audit: “These findings directly shaped our preprocessing pipeline. Mert will now show how we turned those audit decisions into a reproducible pipeline.”
- Mert → Berfin, after Results: “So the next question is the most important one: did the model get the right answers for the right reasons? Berfin will now walk through how we tested that.”
- Berfin → Mert, after Conclusion: “With that boundary clear, Mert will finish with one live prediction using the same saved fine-tuned model.”
- Mert → jury, after Live demo: “The pipeline works, but this output remains a research prototype—not a diagnosis. We are ready for your questions.”

## Tabs and files to prepare before the jury enters

1. Browser tab 1: Home page.
2. Browser tab 2: Data audit → **Distribution & sources**.
3. Browser tab 3: Trust & limits → **Matched intervention**.
4. Browser tab 4: Live demo → **Prepared examples**, with **Lung Opacity · outside-lung attention example** selected.
5. Browser tab 5: Live demo → **Offline fallback**.
6. Keep the four prepared image paths available locally; do not depend on a file picker for the main defense.
7. Visit Live demo once before presenting so the fine-tuned EfficientNetB0 is loaded into Streamlit’s resource cache.
8. Run the selected prepared example once and confirm the fallback screenshot exists at `streamlit_app/assets/live_demo_fallback.png`.

## If time runs over

Use the “30-second version” in every speaker-note file. Skip these items first:

- Preprocessing’s paired-augmentation tab.
- Modelling’s training-evidence tab.
- Data audit’s mask-geometry tab; mention it in one sentence only.
- Trust’s full Grad-CAM explanation; keep the matched recall collapse.
- Live upload; use one prepared example or the fallback screenshot.

Never cut the source-entanglement warning, the 90.3% → 57.0% result, or the non-clinical disclaimer.

## Predictable jury questions

### Is the high score caused by source bias?

We cannot assign every correct prediction to one cause. We can say the labels and sources are entangled, Grad-CAM often highlights non-lung regions, and the matched masking intervention caused a concentrated COVID-recall collapse. Together, these are evidence that source-related shortcuts contribute to performance.

### Does Grad-CAM prove where the disease is?

No. Grad-CAM is a coarse map of influential regions for a prediction. It is not lesion segmentation, and its spatial resolution can spill across mask boundaries. We use it to form a hypothesis, then rely more heavily on controlled input interventions.

### Why did masking not solve the problem?

Masking removes visible non-lung pixels, but masks can retain source-linked geometry; cropping can retain framing; deep features have broad receptive fields; and the training sources remain confounded. Masking is a useful intervention, not a guarantee of invariance.

### What did you try after finding the problem?

Lung ROI crops, pixel masking, lungs-only inputs, mask-only inputs, background-only probes, and masked pooling. None established source invariance. The matched full-image versus lungs-only EfficientNet comparison produced the clearest causal evidence.

### Is the model clinically valid?

No. It has only internal validation on a public, source-confounded dataset. Clinical validity would require source-separated external validation, calibrated probabilities, prospective evaluation, subgroup analysis, and clinical oversight.

### Why show the 92.4% winner if the trust test uses a different model?

They answer different questions. The fine-tuned model is the raw-score winner retained in the defense. The 90.3% → 57.0% result comes from a matched pair where masking is the controlled change, making it the cleaner shortcut-learning test.

### Why not claim the model uses only the background?

Because lungs-only models retain substantial performance. The evidence supports mixed reliance on lung information and non-lung shortcuts, not an all-or-nothing conclusion.
