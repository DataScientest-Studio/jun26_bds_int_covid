# Results

Target time: 3 minutes

## Script

This chart compares representative models on the same held-out test set. The two bars show accuracy and macro F1.

The general pattern is clear. Logistic regression provides a useful baseline, but the image-specific deep-learning models perform substantially better. Transfer learning with EfficientNetB0 reached strong results, and fine-tuning improved it further.

Our strongest current model is the CNN trained from scratch. It achieved 93.8 percent test accuracy and 94.7 percent macro F1. Its COVID recall was 98.7 percent. In simple terms, recall asks: among the images that truly belong to a class, how many did the model correctly identify?

The class table shows that the scratch CNN performs strongly across all four classes, not only on the majority Normal class. That is why the macro F1 result is important. It confirms that performance is not being carried entirely by the largest category.

The confusion matrix shown here is for the fine-tuned EfficientNet model. A confusion matrix compares the true label with the predicted label. The diagonal contains correct predictions, while values outside the diagonal are errors. The remaining difficult distinction is mainly between Lung Opacity and Normal. This is understandable because opacity patterns can be subtle, and the Lung Opacity category is broad.

There are two important cautions when reading this page. First, these metrics describe performance on this dataset's test split, not performance in a hospital or on data from another source. Second, the best score does not automatically tell us whether the model used medically meaningful evidence.

That second caution led to the most important part of our project: interpretability and controlled region experiments.

## What to point at

- Point out the progression from logistic regression to the CNN models.
- Point to both accuracy and macro F1; explain why reporting both matters.
- In the confusion matrix, explain the diagonal before discussing the mistakes.

## Transition

The scores are strong, but we still need to ask whether the models succeeded for the right reason.

