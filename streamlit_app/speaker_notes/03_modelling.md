# Modelling — Mert

Target: 2 minutes 15 seconds

## Full script

“This table is the experiment map. Each row has one purpose. The dummy baseline sets a 48.3% majority-class floor. Logistic regression reaches 71.4%, showing that simple pixel statistics already carry signal. The Simple CNN reaches 82.0%, so learned spatial features add value.

Base EfficientNet reaches 87.1% using pretrained visual features. Fine-tuning raises that to 92.4%, the highest raw score among the models retained in this defense.

The remaining rows ask a different question. Lung ROI and lungs-only inputs restrict access to background. Background, mask-only, and masked-pooling probes test whether non-lung pixels or mask geometry remain predictive. These are diagnostic experiments, not just leaderboard entries.

Every main comparison uses the same deduplicated split and reports accuracy, macro F1, per-class recall, and confusion matrices.”

## Point at

- Rows 1–5 quickly.
- Pause on row 5, the raw-score winner.
- Pause on rows 7–8, which lead into the trust analysis.
- Do not open **Training evidence** unless asked.

## 30-second version

“We moved from dummy and logistic baselines to a Simple CNN and EfficientNet. Fine-tuning produced the best retained raw score. ROI, lungs-only, background, mask-only, and masked-pooling variants then tested where the predictive signal came from.”

## Transition

“First I will show the winner on the internal test split; then I will challenge whether that win is trustworthy.”
