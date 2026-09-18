# Live demo

Target time: 2 minutes 30 seconds

## Before presenting

- Keep one known frontal chest X-ray ready in an easy-to-find folder.
- Run one prediction with each model before the defense so TensorFlow and the models are already initialized.
- Use the fine-tuned EfficientNetB0 model shown in the presentation.
- Keep a screenshot of a successful result as a fallback in case the live environment fails.

## Script before uploading

This final page lets us choose between a Simple CNN, the original EfficientNetB0 transfer model, its fine-tuned version, and two modes for the materialized EfficientNetB0. The scan-only mode accepts the X-ray directly. The scan-and-mask mode also accepts a matching lung mask and reproduces the model's training-time background removal.

The uploaded image is converted to grayscale and resized to the input size expected by the selected model. The model then returns one probability for each of the four classes. These probabilities express the model's relative confidence across its available choices. They are not clinical probabilities and they do not express the chance that a patient has a disease.

*Select the model, upload the prepared images, and click Run predictions.*

## Script after the result appears

For each image, the model predicts **[say the displayed class]** with **[say the displayed confidence]** confidence.

The bar chart is important because it shows all four outputs, not only the winning class. If two bars are close, the prediction is less decisive. If one bar is much larger, the model strongly prefers that class within these four choices.

Below that, the Grad-CAM overlay shows the regions that most influenced this prediction. Warmer colors represent stronger influence. I would not interpret this as the location of a lesion. Grad-CAM is a coarse explanation of model behavior, and our earlier analysis showed that it can also highlight background regions.

This demo shows that the complete batch pipeline works—from multi-image upload and preprocessing to inference and explanation. But the safety message remains the same: these are educational models evaluated on a public dataset, not medical diagnosis systems.

## Closing statement

To conclude, we built a strong four-class chest X-ray classifier, but the most valuable result was learning not to trust a high score without testing where it comes from. Our project combines performance with bias analysis, controlled experiments, and honest limitations. Thank you, and I am happy to answer your questions.

## If the demo is slow

Say: “The first prediction loads the TensorFlow model into memory. Once loaded, the application caches it, so later predictions are faster.”

## If the prediction is wrong

Do not apologize or hide it. Say: “This is a useful example of why a single prediction and a high average test score should not be treated as clinical certainty. Let us look at the full probability distribution and the Grad-CAM region.”
