# Results — Mert

Target: 2 minutes 15 seconds

## Full script

“The retained raw-score winner is fine-tuned EfficientNetB0: 92.4% test accuracy, 93.6% macro F1, and 97.8% COVID recall. Macro F1 gives equal weight to all four classes, so the result is not carried only by the large Normal class.

This is the only confusion matrix we keep in the presentation. The diagonal shows correct predictions; the largest remaining ambiguity is between Lung Opacity and Normal. The adjacent table gives precision, recall, and F1 for each class.

The box at the bottom is the transition. These metrics answer which retained model scores highest on this dataset split. They do not answer whether its evidence is clinically meaningful or stable across sources.”

## Point at

- The three top metrics.
- The diagonal of the confusion matrix.
- The Lung Opacity and Normal rows.
- The “Raw score ≠ trustworthiness” box.

## 30-second version

“Fine-tuned EfficientNetB0 wins with 92.4% accuracy and 93.6% macro F1. Its class-level results are strong, but this internal score cannot tell us what evidence the model used or whether it will survive a source shift.”

## Transition

“That is why our next experiment asks whether the models succeeded for the right reason.”
