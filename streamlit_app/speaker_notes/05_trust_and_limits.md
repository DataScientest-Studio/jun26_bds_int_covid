# Trust and limitations — Berfin

Target: 4 minutes

## Before speaking

Start on **Grad-CAM evidence**, move briefly through **What we tried**, then spend most of the time on **Background vs lungs** and **Resolution**. Finish on **Limitations**. Everything is on one page—no second browser tab needed.

## Full script

“To understand what the model was using, we first looked at Grad-CAM.

In many images, the highlighted areas spread to the borders, shoulders, or background—not just the lungs. For our best model on COVID images, 24.3% of the attention falls inside the lungs, while the lungs cover 24.4% of the image. In other words, it looks at the lungs no more than chance would predict.

That was a warning sign, but not proof. Grad-CAM is only a rough map; it cannot prove cause and effect. So we intervened.

*(Open **What we tried**.)*

We tested lung crops, pixel masks, lungs-only images, mask-only images, and masked pooling. The clearest result is at the top of the page. When we kept only the lungs and blacked out everything else, COVID recall fell from 90.3% to 57%—a drop of 33 points, far more than for any other class.

*(Open **Background vs lungs**.)*

That experiment removed the background. So we also ran the reverse: we blacked out the lungs and kept everything else. If the model were really reading lung disease, this version should do badly—it can’t see a single lung pixel.

We trained each model three ways: on the full image, on the lungs only, and on the background only. The scores here are macro F1. In every architecture—from a small network trained from scratch to the pretrained EfficientNet—the background-only model beats the lungs-only model. For EfficientNet, the model that never sees the lungs scores 0.876; the model that sees only the lungs scores 0.823. Removing the lungs costs just 0.03 compared with the full image.

*(Open **Resolution**.)*

Next, we checked whether the model needs detail. We shrank the images to 16 by 16 pixels, where the lungs are about 8 by 8 pixels—far too small to show any disease pattern. Macro F1 did not drop: 0.835, against 0.791 at full resolution. And at 16 by 16 with the lungs blacked out, it still reaches 0.787. For COVID the gap is clearest: an F1 of 0.67 from the background, but only 0.53 from the lungs.

This connects straight back to the data audit. COVID images come from COVID-only repositories, so recognising the source is enough to recognise COVID.

*(Open **Limitations**.)*

We need to phrase this carefully. It does not mean the model uses only the background—lungs-only models still perform well above chance. It means a large share of the score comes from source-related cues. And our study has limits: we had no data from new hospitals, each experiment was run once, the masks themselves come from a model, and memory limits meant the background test used a frozen backbone.”

## Point at

- Borders and shoulders in the Grad-CAM panel.
- 90.3% → 57.0% in the metrics at the top.
- The Background and Lungs columns, then the Bg − Lungs column: positive in every row.
- The three 16 × 16 metrics.
- The warning about careful wording.

## 30-second version

“Grad-CAM showed attention spreading outside the lungs, so we tested it. Keeping only the lungs cut COVID recall from 90% to 57%. Then we did the reverse—blacked out the lungs—and in every model, the background alone beat the lungs alone. Performance even survived 16 × 16 images. That is strong evidence of shortcut learning—not proof that the lungs carry no signal.”

## If asked

- **Why does the table show 0.907 for EfficientNet, not 0.936?** The region comparison uses the frozen-backbone model, so all three conditions are trained identically. Fine-tuning the full backbone ran out of memory on our hardware.
- **Doesn’t masked pooling solve it?** It kept most of the performance, but deep features over the lungs still summarise most of the image. It may hide the shortcut rather than remove it.
- **How can macro F1 stay at 0.823 when COVID recall is only 57%?** Macro F1 averages all four classes. COVID drops sharply, but the other three barely change.

## Transition

“Those experiments define both the strength and the limit of our result. I’ll now summarize what we can claim and what must come next.”
