# Data audit — Berfin

Target: 3 minutes

## Before speaking

Start on the four example X-rays with their masks. Keep **Distribution & sources** open for most of the section—the source table is the heart of it. Then open **Audit findings** and show only the RGB and Duplicates charts. Mention intensity from the table without opening that chart.

## Full script

“Here is one example from each of the four classes, and under each one, its lung mask. The masks come with the dataset. They were produced by a segmentation model, so we treat them as a tool rather than perfect ground truth. Later we use them to show a model only the lungs—or everything except the lungs.

The dataset has 21,165 images, and 21,106 after removing duplicates. It is also imbalanced: almost half the images are Normal, and Viral Pneumonia is only about 6%. So we don’t rely on accuracy alone—we report macro F1 and recall for each class.

But the biggest issue is where the images came from. Look at this table. Every COVID image comes from six repositories that contribute nothing else—no Normal, no Lung Opacity, no Viral Pneumonia. Lung Opacity comes only from RSNA. And Viral Pneumonia comes only from a Kaggle collection of children’s X-rays.

Here is how strong that link is. If you ignore the image completely and only know which repository it came from, you already identify every single COVID case—and you get 65% accuracy overall, without looking at a single pixel.

So if the source leaves any visible trace in the image—a border, a scanner style, compression, even the patient’s age—the model can use that trace instead of the lungs.

From the audit we drew four findings, each with an action.

The first is this source–label link. It motivates the experiments we show later.

Second, all 140 RGB images were Viral Pneumonia, so we converted every image to grayscale.

Third, 59 files were duplicates, mostly COVID. We removed them before splitting the data.

And fourth, brightness and contrast differ between classes—one more reason to test lung-restricted inputs.”

## Point at

- Each X-ray and the mask beneath it.
- The largest and smallest class bars.
- “Exclusive — no other class” in the COVID row of the source table.
- The two source-only metrics: COVID F1 1.00 and 65.3% accuracy.
- Each row of the finding → action table; open only the RGB and Duplicates charts.

## 30-second version

“The dataset is imbalanced, but the bigger problem is its sources: every COVID image comes from repositories that contain nothing else, so the source alone identifies every COVID case. We converted all images to grayscale, removed 59 duplicates before splitting, and designed experiments to test what the model really uses.”

## If asked

- **Where do the source numbers come from?** From the metadata distributed with the dataset, which records the URL of each image’s original collection.
- **Why ‘children’s X-rays’?** The Kaggle collection comes from a paediatric hospital. Part of the Normal class comes from the same collection.

## Handoff to Mert

“These findings directly shaped our preprocessing pipeline. Mert will now show how we turned those audit decisions into a reproducible pipeline.”
