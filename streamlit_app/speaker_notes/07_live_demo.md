# Live demo — Mert

Target: 2 minutes 30 seconds

## Before presenting

- Visit the page once so the primary model enters the resource cache.
- Keep **Prepared examples** open with **Lung Opacity · outside-lung attention example** selected.
- Run it once before the jury arrives.
- Keep a second browser tab on **Offline fallback**.
- Do not use the upload path unless the jury asks.

## Full script

“For this demo, we’re using our fine-tuned EfficientNetB0—the same model we showed in the results.

We prepared one example from each class so the demo doesn’t depend on uploading a file. I’ve chosen this Lung Opacity image because we already know its Grad-CAM extends outside the lungs. It lets us show the limitation honestly instead of choosing only a perfect-looking example.

The model will give us a score for all four classes. Let’s run it.”

After the result:

“The model predicts **[read label]**, with **[read confidence]** confidence.

Here we can see the scores for all four classes. Showing all four gives us more information than only showing the top result. But these are model outputs—they are not the real probability that this patient has a disease.

On the right is the Grad-CAM view. The warmer colors show the areas that influenced the prediction most. They do not show a lesion. And here, some of the attention is outside the lungs, which matches the shortcut concern we discussed earlier.

So the full pipeline works, from image input to prediction and explanation. But it is still a research prototype, not a diagnostic system.”

## Point at

- The primary-model badge.
- The predicted label.
- All four probability bars.
- The warm region outside the lungs.
- The non-clinical warning.

## 30-second version

“This model gives us all four class scores and a Grad-CAM explanation. The heatmap can focus outside the lungs, which reinforces the limitation we found earlier. The demo works, but it is not a diagnostic tool.”

## If live inference fails

Open **Offline fallback** and say: “The live model isn’t available right now, so I’ll use the saved result. We can still see the same three outputs: the predicted class, all four scores, and the Grad-CAM view. The interpretation stays the same.”

## If the prediction is wrong

“This is a useful example of why one prediction—and even a high average score—doesn’t give us clinical certainty. A wrong result is something we need to understand, not hide.”

## Closing line

“The pipeline works, but this output remains a research prototype—not a diagnosis. We are ready for your questions.”
