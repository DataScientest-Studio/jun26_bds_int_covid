# Dataset and question

Target time: 2 minutes

## Script

Our dataset contains 21,165(twenty-one thousand one hundred and sixty-five) chest X-ray images. Every image is 299 by 299 pixels, and each image also has a lung segmentation mask. The four classes are COVID-19, Lung Opacity, Normal, and Viral Pneumonia.

The chart shows that the classes are not balanced. Normal represents about 48 percent of the dataset, while Viral Pneumonia represents only about 6 percent. We statistically confirmed that this was not a small variation from an equal distribution. The chi-square test was highly significant, and the effect size was large.

This imbalance changes how we evaluate the models. If we only report accuracy, a model can look good mainly because it performs well on the largest class. That is why we also use macro F1, class-specific recall, and confusion matrices. Macro F1 is especially useful here because it gives the four classes equal weight.

The second issue is the origin of the data. These classes were assembled from different public sources. For example, Lung Opacity came from one main source, while Viral Pneumonia came from another. COVID images were collected from several repositories.

This creates a risk: disease class and image source are partly connected. A model might learn differences in scanners, borders, image quality, compression, or hospital-specific processing instead of learning only disease patterns in the lungs. This is called shortcut learning.

So our challenge was not simply to obtain the highest possible score. We also needed to test what information the model was actually using.

## What to point at

- Point to the large Normal bar and the small Viral Pneumonia bar.
- Point to the four class names while introducing the task.
- Briefly point to the statistical-validation text; do not spend time explaining the chi-square formula.

## Transition

With those risks in mind, we treated exploration as a bias audit rather than only looking at example images.

