# Results — Mert

Target: 2 minutes 15 seconds

## Full script

“Here is our best-performing model: the fine-tuned EfficientNetB0.

It reached 92.4% test accuracy, 93.6% macro F1, and 97.8% recall for COVID. Macro F1 is important here because it gives equal weight to all four classes. So the large Normal class is not carrying the result by itself.

This confusion matrix shows where the predictions were correct and where the model still struggled. Most values are on the diagonal, which is what we want. The main confusion is between Lung Opacity and Normal. The table next to it gives the detailed score for each class.

So, on this internal test set, the result is strong. But these numbers only tell us how well the model performed on this split. They don’t tell us what the model looked at, or whether it would work with images from a different hospital or source.”

## Point at

- The three top metrics.
- The diagonal of the confusion matrix.
- The Lung Opacity and Normal rows.
- The “Raw score ≠ trustworthiness” box.

## 30-second version

“The fine-tuned EfficientNetB0 was our best model, with 92.4% accuracy and 93.6% macro F1. The scores are strong across the classes, but they still don’t tell us what the model learned or whether it would work on data from a new source.”

## Handoff to Berfin

“So the next question is the most important one: did the model get the right answers for the right reasons? Berfin will now walk through how we tested that.”
