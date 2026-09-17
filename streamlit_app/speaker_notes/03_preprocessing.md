# Preprocessing

Target time: 2 minutes

## Script

This page summarizes how we transformed the raw collection into reproducible model inputs.

We started with 21,165 images. The first operation was to remove the 59 redundant duplicates. This happened before splitting the data, which is important because it prevents the same image from entering both training and evaluation sets. After this step, 21,106 images remained.

Next, every X-ray was converted to grayscale so that RGB encoding could not act as a class shortcut. Images were resized for the model, and the lung masks were aligned with them. The masks were resized using nearest-neighbor interpolation, because this preserves their binary boundaries instead of creating artificial gray values.

We then created stratified splits: 70 percent for training, 15 percent for validation, and 15 percent for final testing. Stratification means that each split keeps approximately the same class proportions. We used a fixed random seed of 42 so the experiment can be reproduced.

Some choices were always applied, such as deduplication, grayscale conversion, and the fixed split. Other choices were treated as experiments. We compared full images, lung-centered crops, lungs-only inputs, and background-only inputs. This allowed us to measure where predictive information was coming from.

We also tested augmentation. Small rotations were useful as a controlled variation. Horizontal flipping reduced performance, probably because chest anatomy is not perfectly left-right interchangeable, so it was removed from the preferred augmentation setup.

The overall principle was simple: each transformation had to address a problem found during exploration, and experimental alternatives had to remain comparable.

## What to point at

- Follow the pipeline badges from left to right.
- Use the transformation figure to distinguish the original image, the mask, and the processed model input.

## Transition

Once the data pipeline was stable, we increased model complexity step by step.

