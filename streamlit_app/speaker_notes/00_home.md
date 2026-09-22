# Home — Berfin

Target: 45 seconds

## Full script

“Hi everyone. Our project is about classifying chest X-rays into four groups: COVID, Lung Opacity, Normal, and Viral Pneumonia.

We adressed two questions. First, performance: how well can a neural network do this classification? Our best model reached a macro F1 of 0.936 which is on par with published results on this dataset.

Second, attribution: what is the model actually using when doing the classification? Is it reading the lungs, or is it recognising the properties of the resource that the images came from? As you’ll see, a large part of the answer is the second one. So we identified shortcut learning in this project.


## If asked

- **Why macro F1 and not accuracy?** Almost half the images are Normal. Accuracy is dominated by that class; macro F1 gives the small classes the same weight as the big ones. Accuracy for the same model is 92.4%.

## Transition

“I’ll start with the data audit, because the structure of the dataset explains a lot about our findings that comes later.”
