# Trust and limitations — Mert

Target: 4 minutes

## Before speaking

Keep **Matched intervention** open in a separate browser tab. Start on **Grad-CAM evidence**, move briefly to **What we tried**, and finish on **Matched intervention**.

## Full script

“Grad-CAM frequently highlighted borders, shoulders, and background. Its average lung focus was only mildly above geometric chance, and for one class it fell below chance. This raised a concern, but Grad-CAM is coarse and correlational; it cannot prove causation or locate a lesion.

We therefore tried several interventions: lung ROI crops, pixel masking, lungs-only inputs, mask-only inputs, background-only probes, and masked pooling. These did not establish that the source-related shortcut had disappeared. Crops can retain framing, masks can retain source-linked geometry, and the underlying sources remain confounded.

The cleanest test is this matched EfficientNet pair. Architecture, split, and seed stayed fixed; the intervention removed access to pixels outside the lungs. COVID recall fell from 90.3% to 57.0%, a 33.3-point collapse, while the other classes fell much less.

This is evidence that the full-image model exploited COVID-specific information outside the lungs. It is not proof that the model uses only background: the lungs-only model still retains substantial performance, so the model learned a mixture of lung signal and shortcut signal.

The 92.4% winner on the Results page and this matched pair answer different questions. The winner answers raw performance; the matched pair provides the cleaner causal test.”

## Point at

- Borders and shoulders in the Grad-CAM panel.
- The five attempted interventions.
- 90.3%, 57.0%, and −33.3 points.
- The warning about careful wording.

## 30-second version

“Grad-CAM suggested non-lung attention, so we tried crops, masks, lungs-only, mask-only, background, and masked pooling. In the matched test, removing background collapsed COVID recall from 90.3% to 57.0%. This is evidence of shortcut learning, not proof that the model uses only background.”

## Handoff to Berfin

“Those interventions define both the strength and the limit of our result. Berfin will now summarize what we can claim and what must come next.”
