# Conclusion — Berfin

Target: 1 minute 30 seconds

## Full script

“To summarize, there are three main points.

First, we built a strong and reproducible four-class classifier. We used a fixed split, saved the model artifacts, and checked performance for each class.

Second, the main limitation is the dataset itself. The disease labels are closely linked to the image sources. So the final score reflects both useful lung information and dataset-specific shortcuts.

Third, the most important next step is external validation with sources kept separate. Only after that would it make sense to focus on methods like domain harmonization, better lung constraints, probability calibration, or clinical evaluation. These are future steps, not completed results.

So our main takeaway is simple: a good performance score is an engineering result. Trusting the model is a separate scientific question.”

## Point at

- “What worked.”
- “Main limitation.”
- The first future-work item: source-separated external validation.
- “Future work—not completed claims.”

## 30-second version

“We built a strong and reproducible classifier, but the labels are closely connected to the data sources. The next step must be external validation with separated sources. Everything beyond that—better constraints, calibration, and clinical testing—is still future work.”

## Handoff to Mert

“With that boundary clear, Mert will finish with one live prediction using the same saved fine-tuned model.”
