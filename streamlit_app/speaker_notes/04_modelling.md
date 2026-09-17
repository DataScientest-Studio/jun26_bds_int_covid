# Modelling

Target time: 2 minutes 30 seconds

## Script

We did not begin with the most complex neural network. We created a sequence of models so that each step had a clear comparison.

The first level was a dummy classifier and logistic regression. The dummy classifier gives us the trivial floor: what happens if we mainly follow class frequency? Logistic regression is a simple linear model trained on downsampled pixel values. It tells us how much can be achieved without learning complex spatial patterns.

The second level used boosted trees. This provides a nonlinear classical machine-learning benchmark, but it still does not learn images in the same way as a convolutional network.

The third level was convolutional neural networks trained from scratch. A CNN learns spatial filters such as edges, textures, and increasingly complex image patterns directly from our dataset. We tested both simple architectures and a deeper regularized CNN.

The fourth level was transfer learning with EfficientNetB0. This network begins with visual features learned from ImageNet. We first froze the pretrained backbone and trained a new classification head. We then fine-tuned the network using a much smaller learning rate so that its features could adapt to chest X-rays without being destroyed by large updates.

All main comparisons used the same deduplicated, stratified data split. We evaluated on an untouched test set using accuracy, macro F1, class recall, and confusion matrices. This makes the comparison much fairer than allowing every model to use a different test sample.

We also tested class weighting, balanced sampling, augmentation, lung masking, lung cropping, and background-only inputs. These were not simply attempts to increase one score. Some experiments were designed to understand the model and expose possible confounding information.

## A simple explanation if asked

- A CNN learns its filters from this dataset.
- Transfer learning starts from filters learned previously on a very large image dataset.
- Fine-tuning updates some or all of those pretrained filters using a low learning rate.

## Transition

Now I will compare the held-out results and show which approach performed best.

