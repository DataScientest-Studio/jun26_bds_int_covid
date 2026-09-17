# Interpretability and limits

Target time: 4 minutes

## Script

This is the central finding of the project. A high-performing medical-image model is only useful if its decisions are based on relevant anatomy rather than accidental dataset clues.

*Keep the Visual evidence tab open.*

We first used Grad-CAM. Grad-CAM produces a coarse heatmap showing which regions influenced a prediction. Warmer colors indicate stronger influence. On the examples shown here, the activation often extends toward borders, shoulders, and corners rather than staying clearly inside the lungs.

We then quantified this instead of relying only on visual impressions. For each image, we measured the share of Grad-CAM activation inside the lung mask. We compared that with a chance level: the fraction of the image occupied by the lungs. If model attention is close to this chance value, it is not especially lung-focused.

For three classes, lung focus was only mildly above chance. For Lung Opacity, it was below chance. This suggested a problem, but Grad-CAM is correlational and spatially coarse. It cannot prove by itself that background information caused the model's performance.

*Open the Causal test tab.*

To test this more directly, we trained an otherwise comparable EfficientNet model after setting every pixel outside the lung masks to zero. The model could no longer access borders, labels, shoulders, or other background information.

Overall test accuracy fell by 6.6 percentage points. The most important result was not the average drop but how uneven it was. COVID recall fell from 90.3 percent to 57.0 percent, a loss of 33.3 percentage points. Performance for the other classes fell much less.

If the background contributed equally to every class, we would expect a more even decline. Instead, the damage was concentrated in COVID. This is direct experimental evidence that the original model used COVID-specific information outside the lungs.

This does not mean that the entire model is useless or that it learned nothing from lung tissue. It means that the excellent in-dataset score combines real lung information with dataset-specific shortcuts. A new hospital with different scanners, borders, or image-processing conventions may not contain the same shortcuts, so performance—especially COVID recall—could collapse.

Our conclusion is therefore deliberately cautious. The model is a successful educational prototype and a strong demonstration of the complete data-science workflow. It is not ready for clinical use. The next scientific step would be external validation on source-separated hospital data, followed by calibration and prospective clinical evaluation.

## Important wording

Say **“evidence of background shortcut learning”**, not **“proof that the model diagnoses from borders only.”** The model still retains useful performance with lungs-only inputs.

Say **“Grad-CAM shows influential regions”**, not **“Grad-CAM shows the lesion.”** It is not a segmentation method.

## Transition

With those limitations clearly stated, I will now demonstrate how one of the saved models processes a new image.

