# Preprocessing — Berfin

Target: 1 minute 30 seconds

## Full script

“We first removed 59 redundant files, leaving 21,106 images. Only then did we create the fixed, stratified 70/15/15 split with seed 42. Every image was converted to grayscale and resized. Masks used nearest-neighbor resizing so their binary boundaries stayed intact.

The full-image pipeline is the default comparison. Lung ROI, lungs-only, background-only, and other constrained inputs remain explicit experiments. Small rotations were retained; horizontal flipping was removed after it hurt validation performance. When an image and mask are augmented, the same transform is applied to both.”

## Point at

- Follow the five pipeline badges from left to right.
- If time allows, open **Paired augmentation** and point to alignment.

## 30-second version

“We deduplicated before a fixed stratified split, standardized every image to grayscale, aligned masks safely, and kept region constraints as experiments. This made all later comparisons reproducible.”

## Handoff to Mert

“The audit told us what to standardize and what to test. Mert will now show the experiment sequence we used to separate raw performance from trustworthy evidence.”
