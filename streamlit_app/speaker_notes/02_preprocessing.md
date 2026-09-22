# Preprocessing — Mert

Target: 1 minute 30 seconds

## Full script

“This is the full preprocessing pipeline.

We started by removing the 59 redundant files, which left us with 21,106 images. Then we created a fixed 70-15-15 split for training, validation, and testing. We used a stratified split, so each part keeps a similar class balance.

After that, every image was converted to grayscale and resized. We resized the masks with nearest-neighbor interpolation, so the mask boundaries stayed clean.

The full image is our main input. The lung crop, lungs-only, and background-only versions are separate experiments, so we can compare them fairly.

For augmentation, we kept small rotations. We also tested horizontal flips, but removed them because validation performance became worse. And whenever we rotate an image, we apply exactly the same rotation to its mask, so they stay aligned.”

## Point at

- Follow the five pipeline badges from left to right.
- If time allows, open **Paired augmentation** and point to alignment.

## 30-second version

“We removed duplicates before making a fixed, stratified split. Then we converted the images to grayscale, resized the images and masks safely, and kept the lung-focused inputs as separate experiments. This gave us a consistent pipeline for every model comparison.”

## Transition

“The pipeline is consistent across every model comparison. Next I’ll show the experiment sequence we used to separate raw performance from trustworthy evidence.”
