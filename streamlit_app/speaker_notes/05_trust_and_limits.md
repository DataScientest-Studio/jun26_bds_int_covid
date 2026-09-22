# Trust and limitations — Berfin

Target: 4 minutes

## Before speaking

Keep **Matched intervention** open in a separate browser tab. Start on **Grad-CAM evidence**, move briefly to **What we tried**, and finish on **Matched intervention**.

## Full script

“To understand what the model was using, we first looked at Grad-CAM.

In many images, the highlighted areas spread to the borders, shoulders, or background—not just the lungs. For Lung Opacity, the average focus inside the lungs was even below what we would expect by chance.

That was a warning sign, but it wasn’t proof. Grad-CAM is only a rough map of which areas influenced a prediction. It cannot show us a lesion, and it cannot prove cause and effect.

So we went further. We tested lung crops, pixel masks, lungs-only images, mask-only images, background regions, and masked pooling. Each experiment removed or restricted a different part of the image.

None of these tests showed that the shortcut was completely gone. A crop can still keep the original framing, and a mask can still carry patterns from the source or annotation process.

The clearest result came from this matched comparison. We kept the EfficientNet architecture, data split, and random seed the same. The main change was removing the pixels outside the lungs.

When we did that, COVID recall dropped from 90.3% to 57%. That is a fall of 33.3 percentage points, and it was much larger than the change for the other classes.

This gives us evidence that the full-image model was using COVID-related information from outside the lungs.

But we need to phrase that carefully. It does not mean the model used only the background. The lungs-only model still performed reasonably well. The model seems to use a mixture of real lung information and shortcut information.

And one final distinction: the 92.4% model on the previous page is our best raw-score model. This matched comparison is a separate experiment designed to test the effect of removing the background. They answer two different questions.”

## Point at

- Borders and shoulders in the Grad-CAM panel.
- The five attempted interventions.
- 90.3%, 57.0%, and −33.3 points.
- The warning about careful wording.

## 30-second version

“Grad-CAM suggested that the model was looking outside the lungs, so we tested several restricted versions of the input. In the cleanest comparison, removing the background reduced COVID recall from 90.3% to 57%. That is strong evidence of shortcut learning, but it does not mean the model uses only the background.”

## Transition

“Those interventions define both the strength and the limit of our result. I’ll now summarize what we can claim and what must come next.”
