# Home — Berfin

Target: 45 seconds

## Full script

“Hi everyone. Our project is about classifying chest X-rays into four groups: COVID, Lung Opacity, Normal, and Viral Pneumonia.

We asked two questions. First, performance: how well can a neural network do this? Our best model reached a macro F1 of 0.936—a score that gives all four classes equal weight—which is on par with published results on this dataset.

Second, attribution: what is the model actually using? Is it reading the lungs, or is it recognising which repository an image came from? As you’ll see, a large part of the answer is the second one.

And to be clear from the start: this is a research prototype, not a clinical diagnostic tool.”

## Point at

- The two research questions.
- “Best model: 0.936 macro F1.”
- “Shortcut learning identified.”

## 30-second version

“We classified chest X-rays into four groups and asked two questions: how well does the model perform, and what is it actually using? It scores well—0.936 macro F1—but much of that comes from shortcuts in how the dataset was built. It is a research prototype, not a clinical tool.”

## If asked

- **Why macro F1 and not accuracy?** Almost half the images are Normal. Accuracy is dominated by that class; macro F1 gives the small classes the same weight as the big ones. Accuracy for the same model is 92.4%.

## Transition

“I’ll start with the data audit, because the structure of the dataset explains almost everything that comes later.”
