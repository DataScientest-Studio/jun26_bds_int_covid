# Conclusion — Berfin

Target: 1 minute 30 seconds

## Full script

“To summarize, let’s go back to our two questions.

On performance: transfer learning clearly helps. A small network trained from scratch reached 0.827 macro F1, a frozen EfficientNet 0.907, and after fine-tuning 0.936—on par with published results on this dataset.

On attribution: a large share of that performance does not come from the lungs. Every model does better on the background than on the lungs, and even a 16-by-16 thumbnail with the lungs removed reaches 0.787.

So the next step is clear: test on images from hospitals that are not in this dataset. Our results make a concrete prediction—COVID recall should drop the most. Beyond that, we’d train a model to predict the source directly, penalise source information during training, and, above all, build datasets where each hospital contributes more than one class. These are future steps, not completed results.

Our main takeaway: high accuracy on this dataset does not, by itself, show that a model recognises disease. Where the performance comes from has to be part of the evaluation.”

## Point at

- The Performance card.
- The Attribution card.
- The first future-work item: external validation.
- The final takeaway.

## 30-second version

“Our best model reached 0.936 macro F1, on par with published work. But every model did better on the background than on the lungs, so much of that score comes from dataset shortcuts. The key next step is testing on new hospitals—and the key lesson is that where performance comes from must be part of its evaluation.”

## Handoff to Mert

“With that boundary clear, Mert will finish with one live prediction using the same saved fine-tuned model.”
