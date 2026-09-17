# Exploration

Target time: 3 minutes

## Script

During exploration, we looked for technical properties that could accidentally reveal the label to the model. I will show three examples: encoding, duplicates, and image intensity.

*Open the Encoding tab.*

First, we found that 140 images were stored as RGB images, and every one of them belonged to the Viral Pneumonia class. The other images were grayscale. The plot also shows that these RGB files were larger.

RGB encoding has no direct medical meaning here, but it is associated with one class. If we left it unchanged, the model could use the encoding difference as an easy shortcut. Our response was to convert every image to the same grayscale format.

*Open the Duplicates tab.*

Second, we checked for exact duplicate images. We first used perceptual hashing to find candidates and then confirmed duplicates through pixel-level comparison. We identified 59 redundant files. Most of these were in the COVID class.

The total percentage is small, only 0.28 percent, but duplicates can still cause data leakage. If identical images appear in both training and test data, the test score becomes too optimistic. We therefore removed the redundant copies before making any train, validation, or test split.

*Open the Intensity tab.*

Third, we compared brightness and contrast across the four classes. COVID images had the highest average brightness, while Normal images had the highest average contrast. These differences may partly come from acquisition or preprocessing rather than disease.

The important point is that we did not treat these plots as interesting observations only. Each finding produced an action: standardize image encoding, remove duplicates before splitting, and later compare full-image models with lung-focused models to test whether global image characteristics were influencing predictions.

## What to emphasize

- A technical correlation can be predictive without being medically meaningful.
- Removing only 59 duplicates still matters because the goal is an honest test set.
- Brightness differences motivated experiments; we did not assume that brightness itself represented disease.

## Transition

These exploration findings directly determined our preprocessing pipeline.

