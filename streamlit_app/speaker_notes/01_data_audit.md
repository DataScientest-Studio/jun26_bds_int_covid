# Data audit — Berfin

Target: 3 minutes

## Before speaking

Keep **Distribution & sources** open. During the section, open **Three audit findings**, then show only its RGB and Duplicates charts. Mention intensity from the mapping table without opening that chart. Do not narrate the mask chart unless asked or time allows.

## Full script

“Here we have one example from each of the four classes. In total, the dataset contains 21,165 images.

The first thing we noticed was the class imbalance. Almost half of the images are Normal, while Viral Pneumonia makes up only around 6%. So later, we don’t rely on accuracy alone. We also look at macro F1 and recall for each class.

But the bigger issue is where the images came from. Lung Opacity comes from one source, Viral Pneumonia comes from another, and the other two classes use a mix of sources. This means the disease label and the data source are connected.

Why does that matter? The model might learn things like borders, scanner style, compression, or preprocessing—instead of learning only from the lungs.

We found three practical problems and made one decision for each of them.

First, all 140 RGB images were in the Viral Pneumonia class, so we converted every image to grayscale.

Second, we found 59 duplicate or redundant files, mostly in the COVID class. We removed those before splitting the data.

Third, brightness and contrast were different across the classes. That gave us another reason to test lung-focused versions of the images later.

We also checked the lung masks. Their size and shape vary by class, so even a mask can carry information about the data source. That’s why we treat masking as an experiment, not as an automatic solution to bias.”

## Point at

- One X-ray from each class.
- The largest and smallest class bars.
- The “RSNA only” and “Kaggle only” rows.
- Each row of the finding → action table. Explain only the class-distribution, RGB, and duplicate charts; leave intensity and mask geometry as supporting visuals.

## 30-second version

“The dataset is imbalanced, and the classes are connected to different data sources. We converted all images to grayscale, removed 59 duplicates before the split, and used the brightness differences as a reason to test lung-focused inputs. We also found that the masks are not automatically free from source bias.”

## Handoff to Mert

“These findings directly shaped our preprocessing pipeline. Mert will now show how we turned those audit decisions into a reproducible pipeline.”
