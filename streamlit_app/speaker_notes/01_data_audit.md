# Data audit — Berfin

Target: 3 minutes

## Before speaking

Keep **Distribution & sources** open. During the section, open **Three audit findings**, then show only its RGB and Duplicates charts. Mention intensity from the mapping table without opening that chart. Do not narrate the mask chart unless asked or time allows.

## Full script

“These are representative X-rays from the four target classes. The collection contains 21,165 images, but it is not balanced: Normal is almost half, while Viral Pneumonia is about 6%. That is why accuracy alone is not enough; later we report macro F1 and class recall.

The more important issue is the source table. Lung Opacity comes only from RSNA, Viral Pneumonia only from the Kaggle pneumonia source, Normal is mainly RSNA plus Kaggle, and COVID aggregates several repositories. Disease label and acquisition source therefore move together. A model can use scanner, border, compression, or preprocessing cues as a proxy for disease.

We turned three audit findings into direct actions. First, all 140 RGB files belong to Viral Pneumonia, so every input is converted to grayscale. Second, we found 59 redundant files, concentrated in COVID, so duplicates are removed before any split. Third, brightness and contrast differ by class, so we later test lung-constrained inputs instead of assuming global intensity is pathology.

The supplied masks also differ geometrically by class. That is supporting evidence: masks can carry source or annotation conventions, so they are experimental tools—not guaranteed bias removal.”

## Point at

- One X-ray from each class.
- The largest and smallest class bars.
- The “RSNA only” and “Kaggle only” rows.
- Each row of the finding → action table. Explain only the class-distribution, RGB, and duplicate charts; leave intensity and mask geometry as supporting visuals.

## 30-second version

“The dataset is imbalanced and source-confounded. Three audit findings became actions: class-linked RGB encoding led to grayscale conversion, 59 duplicates led to pre-split deduplication, and intensity differences led to lung-constrained experiments. Mask geometry provided supporting evidence that masks are not automatically neutral.”

## Transition

“Those findings were not left as observations; they became the preprocessing pipeline.”
