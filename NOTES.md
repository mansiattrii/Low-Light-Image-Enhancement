# Notes (interview prep)

Plain-language notes written after each milestone: what was built, why, what was rejected,
and questions I should be ready for. Numbers quoted here come from the files in `results/`.

---

## Milestone 1: setup, data and classical baselines

### What was built

- `src/data.py`: finds LOL pairs by matching filenames in `low/` and `high/`, loads images as
  float32 RGB in [0, 1], and creates the validation split.
- `src/classical.py`: gamma correction, histogram equalization (HE) on luma, CLAHE on the L
  channel of LAB, plus a grid search on the validation split (`python -m src.classical`).
- `src/metrics.py`: one PSNR/SSIM function used by both tuning and evaluation.
- `src/evaluate.py`: scores every method on eval15 (or the validation split) and writes
  `results/results.md`, `results/results.csv` and per-image scores.
- Tests in `tests/` that use small synthetic images, so they run without the dataset.

### Data split

- LOL v1 layout matched the expected `data/LOL/our485/{low,high}` and `data/LOL/eval15/{low,high}`:
  485 and 15 PNG pairs, all 600x400, identical filenames in `low/` and `high/`.
- 35 of the 485 training pairs are held out for validation (about 7%), sampled with seed 42.
- The validation filenames are written to `data/splits/val.txt` and committed. Once that file
  exists it is the source of truth, so the split cannot silently change if the code or seed
  changes. Anyone can see exactly which images were held out.
- eval15 is never used to pick anything. It is only scored at the end.

### Why the methods are implemented this way

**Gamma correction.** `out = in^gamma` per pixel. With gamma < 1 the curve lifts dark values much
more than bright ones (0.01 becomes 0.32 at gamma 0.25, while 0.9 only becomes 0.97). It is a
single global curve: every pixel with the same value gets the same output, whatever is around it.
The gamma value is chosen by grid search on validation (selected: see `results/tuned_params.yaml`).

**Histogram equalization on luma.** HE remaps intensities so the cumulative histogram becomes a
straight line, which spreads the values across the full range. I apply it only to the Y (luma)
channel of YCrCb and keep the colour channels. Equalizing R, G and B separately would change their
ratios and produce colour shifts. HE has no parameters to tune.

**CLAHE on L in LAB.** Contrast-Limited Adaptive HE splits the image into tiles, equalizes each
tile, and blends neighbouring tiles with bilinear interpolation to avoid visible seams. The clip
limit caps how tall any histogram bin can get before equalizing, which limits how much it can
stretch flat regions (and therefore noise). Tuned: clip limit and tile grid size.

**Tuning criterion: mean validation PSNR.** PSNR is the headline metric in the Zero-DCE paper
and in most LOL tables, so tuning on it makes the comparison fair. Both PSNR and SSIM are recorded
for every grid point in `results/tuning_classical_val.csv`.

**A finding from tuning CLAHE.** PSNR kept preferring larger clip limits and fewer tiles, i.e.
less local adaptation. With a 1x1 tile grid allowed, it won (that is global, clipped HE, no longer
adaptive). I excluded tile size 1 so that the CLAHE row really is CLAHE, and kept the grid edge
documented in `config.yaml`. The lesson: PSNR rewards matching the reference's global brightness,
not local contrast, so a local-contrast method is at a disadvantage on this metric.

### Evaluation choices

- **8-bit quantization before scoring.** Each method's output is rounded to 8 bits before
  PSNR/SSIM, because that is the image a user would actually save, and the ground truth is 8-bit.
  This applies equally to every method.
- **RGB, data range 1.0.** Metrics on the full-resolution RGB image, not only on the Y channel.
  Some papers report Y-channel numbers, which are not directly comparable.
- **SSIM settings.** scikit-image's default is a 7x7 uniform window. I use the original definition
  from Wang et al. (2004): 11x11 Gaussian window, sigma 1.5, population covariance. This is what
  most papers mean by SSIM, and the choice changes the numbers noticeably.
- **Mean ± sample std over images.** With only 15 test images the spread matters as much as the
  mean.
- **Timing.** Only the enhancement call is timed (not reading or writing files), with one warm-up
  call. It is wall-clock time on my laptop CPU; OpenCV may use several threads.
- **"Input (no enhancement)" row.** Gives a floor, so every other row can be read as an
  improvement over doing nothing.

### What the baseline results showed (eval15, `results/results.md`)

- Gamma (0.25) is the best classical method on both mean PSNR and SSIM. HE and CLAHE are close
  to each other and clearly behind.
- Why gamma wins on average: LOL's normal-light image is mostly a brighter version of the same
  scene, so the mapping from low to high is close to one smooth global tone curve. Gamma is
  exactly that kind of curve. HE forces a flat histogram, which real scenes do not have, so it
  over-brightens some regions, washes out others and stretches noise.
- Where gamma fails: per-image scores (`results/results_per_image.csv`) show gamma is worst on
  the darkest inputs (55.png, 23.png), where HE beats it by several dB, and it does nothing useful
  on a relatively bright input (179.png). One fixed exponent cannot fit every exposure level.
  HE adapts to each image's own histogram, which helps on the darkest images.
- This is the motivation for Zero-DCE: it predicts a curve per image and per pixel, instead of
  using one global setting for the whole dataset.

### Alternatives considered and rejected

- **Tuning on eval15.** Rejected: picking parameters on the test set inflates the reported
  numbers (test-set leakage).
- **k-fold cross-validation for tuning.** Unnecessary here: the classical methods have one or two
  parameters and are cheap to evaluate; a single held-out set of 35 images is enough and keeps
  the protocol identical to Zero-DCE checkpoint selection.
- **Gamma per image (e.g. chosen from mean brightness).** That would be a different, adaptive
  method. The baseline is meant to be the simple, standard one.
- **HE on each RGB channel.** Causes colour shifts (see above).
- **CLAHE on the V channel of HSV.** LAB's L is closer to perceived lightness, and the standard
  CLAHE recipe uses LAB.

### Background: why low-light enhancement is hard

1. **Noise.** In a dark photo the signal is weak compared with sensor noise (shot noise and read
   noise). Brightening multiplies signal and noise together, so the noise becomes visible.
2. **Colour shifts.** Sensors and white balance behave differently at low light, and channels can
   clip at different points, so a dark image can have a colour cast that plain brightening keeps or
   amplifies.
3. **Over-exposure.** Dark images often contain bright light sources or windows. A curve that lifts
   the shadows enough can blow out regions that were already bright.
4. **Unknown target.** "Good exposure" is subjective. LOL defines it as one reference photo per
   scene, which is a reasonable choice but not the only correct answer.

### Background: PSNR and SSIM

- **PSNR** = 10·log10(MAX² / MSE). It is a log-scaled mean squared error per pixel, so it is
  dominated by large errors and is very sensitive to global brightness differences. 6 dB higher ≈
  error amplitude halved. Limitation: it treats every pixel independently and ignores structure;
  a slightly too-dark but sharp image can score worse than a blurry one with the right brightness.
- **SSIM** compares local windows on three terms: luminance (means), contrast (standard
  deviations) and structure (correlation), then averages over the image. Values near 1 mean
  similar. Limitation: still a reference-based metric; it depends on window settings, does not
  model colour well when computed per channel, and is not the same as human judgement.
- Both need ground truth, so they cannot score my own phone photos. Those are only for visual
  inspection.

### Likely interview questions

- **Why tune on a validation split instead of eval15?** Choosing hyperparameters on the test set
  means the test score is no longer an unbiased estimate of performance on new images.
- **Why is the split saved to a file?** Reproducibility and transparency: the exact held-out
  images are visible and cannot change between runs.
- **Why apply HE to luma only?** Equalizing channels separately changes R:G:B ratios and shifts
  colours; luma-only changes brightness while keeping chroma.
- **What does the CLAHE clip limit do?** It caps histogram bins before equalizing, which limits the
  slope of the mapping and so limits how much noise in flat regions gets amplified.
- **Why did gamma beat CLAHE?** PSNR mostly rewards matching global brightness, and LOL's
  low→high mapping is close to a global tone curve. CLAHE optimizes local contrast, which PSNR does
  not reward.
- **What is gamma's main weakness?** One exponent for every image. It under-brightens the darkest
  scenes and over-brightens the brighter ones (visible in the per-image scores).
- **Why quantize to 8 bits before computing metrics?** To score the image that would actually be
  saved and match the 8-bit ground truth. It treats all methods the same.
- **Why Gaussian-window SSIM?** That is the original definition. scikit-image's default (7x7
  uniform) gives different numbers, so results would not be comparable with papers.
- **What do mean ± std tell you with 15 images?** The standard deviation across images is large
  compared with differences between methods, so small gaps in mean PSNR should not be
  over-interpreted.
