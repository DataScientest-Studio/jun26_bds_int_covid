# Modelling — Mert

Target: 2 minutes 15 seconds

## Full script

“This table gives us the full experiment story. Each row has a different job.

We began with a dummy model. It simply predicts the largest class and gives us a baseline of 48.3% accuracy.

Logistic regression reached 71.4%. That already tells us there is useful information in the pixels, even with a simple linear model. Then the Simple CNN reached 82%, showing that learned spatial features helped.

Next, we moved to transfer learning. The base EfficientNet reached 87.1%, and after fine-tuning it reached 92.4%. That is our best raw score.

From this point, the goal changes. We’re no longer just asking, ‘Can we improve the score?’ We’re asking, ‘Where is the score coming from?’

So we tested lung crops, lungs-only images, background regions, masks, and masked pooling. These experiments help us check whether the model still finds useful signals outside the lungs.

To keep the comparison fair, we used the same cleaned data split and the same evaluation approach throughout.”

## Point at

- Rows 1–5 quickly.
- Pause on row 5, the raw-score winner.
- Pause on rows 7–8, which lead into the trust analysis.
- Do not open **Training evidence** unless asked.

## 30-second version

“We started with simple baselines, then moved to a CNN and EfficientNet. Fine-tuning gave us the best score at 92.4%. After that, we used lung-only, background, mask, and other constrained versions to test where the model’s information was coming from.”

## Transition

“First, I’ll show how the winning model performed. Then we’ll look at whether we can actually trust that result.”
