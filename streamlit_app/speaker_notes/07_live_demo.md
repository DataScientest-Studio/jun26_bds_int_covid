# Live demo — Mert

Target: 2 minutes 30 seconds

## Before presenting

- Visit the page once so the primary model enters the resource cache.
- Keep **Prepared examples** open with **Lung Opacity · outside-lung attention example** selected.
- Run it once before the jury arrives.
- Keep a second browser tab on **Offline fallback**.
- Do not use the upload path unless the jury asks.

## Full script

“For the defense we use one primary model: fine-tuned EfficientNetB0. Four prepared examples—one per class—remove dependence on the file picker. I selected a known example where the Grad-CAM focus extends outside the lungs, because it demonstrates the limitation rather than hiding it.

The model returns one score for each class. I will run a single prediction now.”

After the result:

“The predicted class is [read label] with [read confidence]. The bar chart shows all four model outputs, which is more informative than only the winner. These are model-relative scores, not the probability that a patient has a disease.

The Grad-CAM overlay shows influential regions. Warm colors do not identify a lesion, and attention outside the lungs is consistent with the shortcut-learning concern. The application works end to end, but it remains a non-clinical research prototype.”

## Point at

- The primary-model badge.
- The predicted label.
- All four probability bars.
- The warm region outside the lungs.
- The non-clinical warning.

## 30-second version

“This cached fine-tuned model outputs all four scores and a coarse Grad-CAM explanation. The heatmap can focus outside the lungs, reinforcing the limitation. This is a working research demo, not a diagnostic tool.”

## If live inference fails

Open **Offline fallback** and say: “The live runtime is unavailable, so I am using the pre-recorded successful result. The same three outputs are visible: predicted class, four scores, and Grad-CAM. The scientific interpretation is unchanged.”

## If the prediction is wrong

“This is exactly why one prediction and a high average score are not clinical certainty. The full score distribution and the shortcut analysis matter more than hiding a failure.”

## Closing line

“The pipeline works, but this output remains a research prototype—not a diagnosis. We are ready for your questions.”
