# Failure cases on eval15

Figure: [figures/failure_cases.png](figures/failure_cases.png). For each method it shows the eval15
image with that method's lowest PSNR. Numbers are from `results_per_image.csv`; brightness is
the mean output intensity in [0, 1].

## gamma on 55.png (11.79 dB; brightness 0.360, reference 0.584)

One exponent for every image. 55.png is one of the darkest inputs and its reference is one of the
brightest, so gamma 0.25 (tuned for the average validation image) leaves it too dark. It also
lifts the sensor noise in the shadows, which shows as green and magenta speckle on the cabinet.

## hist_eq and clahe on 179.png (9.14 dB and 9.42 dB; brightness 0.527 / 0.516, reference 0.284)

The reference here is itself a dark scene: a dark brown wall behind saturated tassels. Both
methods spread intensities over the full range regardless of content, so the wall becomes
light grey and the image is almost twice as bright as the reference. Colours also wash out:
equalization changes only luma (Y or L) and keeps the chroma values of the dark input, and those
are small in a dark image, so the brightened result is nearly grey.

## zerodce (official losses, default) and zerodce_paper on 179.png (9.48 dB / 8.75 dB; brightness 0.603 / 0.627, reference 0.284)

The same scene breaks Zero-DCE for a related reason. The exposure loss pulls every 16x16 patch
towards E = 0.6, so a scene whose correct appearance is dark is brightened to roughly twice the
reference. The colour constancy loss assumes the image averages to grey, which is false for a
brown wall with red and pink objects, so the wall turns pink. Both are assumptions of the
zero-reference losses, not training failures: the model does what its losses ask.

## zerodce_paper_e045 on 55.png (11.22 dB; brightness 0.354, reference 0.584)

With the lower exposure target (0.45) the darkest scenes end up too dark, and the result has a
flat grey haze: the whites of the cabinet are grey and the shadows are lifted. This matches the
low contrast measured for this run in `analysis/checkpoint_drift.md`. Shadow noise is amplified
as in the gamma result.

## Common thread

- Every method fails on scenes whose reference brightness is far from the dataset average
  (very bright 55.png, very dark 179.png). Fixed targets (one gamma, one exposure level E, a flat
  histogram) cannot represent "how bright should this particular scene be".
- Noise in the darkest regions is amplified by every method (visible on 55.png and 23.png). This
  is the starting point for the denoising study in Milestone 4.
