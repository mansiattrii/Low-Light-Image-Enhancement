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

---

## Milestone 2: Zero-DCE model, losses and training code

### What was built

- `src/zerodce.py`: DCE-Net, the light-enhancement curve, and helpers to load weights and to
  enhance a NumPy image.
- `src/losses.py`: the four zero-reference losses and their weighted sum.
- `src/train.py`: training loop with per-epoch validation (zero-reference loss, PSNR, SSIM),
  best/last checkpoints, automatic resume, a CSV log and loss curves.
- `notebooks/train_colab.ipynb`: clones the repo, installs requirements, unzips LOL from Drive and
  calls `python -m src.train`. No training logic in the notebook.
- Tests for the curve, the network, each loss, the crop dataset, and a training smoke test that
  checks checkpoints are written and that a resumed run matches an uninterrupted one.

### How the curve works

`LE(x) = x + α·x·(1 − x)`, applied per pixel and per colour channel, with α in [−1, 1].

- α > 0 brightens, α < 0 darkens, α = 0 leaves the pixel unchanged.
- 0 and 1 are fixed points, and for α in [−1, 1] the output never leaves [0, 1]: no clipping,
  nothing thrown away.
- The curve is monotonic (derivative 1 + α(1 − 2x) ≥ 0), so a brighter input pixel never ends up
  darker than a dimmer one in the same channel: intensity order is preserved.
- It is differentiable, so gradients flow through it into the network.

**Why iterate it 8 times.** One quadratic step can lift a value only so far (at most from x to
2x − x²; 0.05 becomes 0.0975). Applying it repeatedly composes the steps into a much higher-order
curve that can brighten very dark pixels strongly while barely moving bright ones. The paper's
ablation shows a single iteration is clearly worse. Each iteration has its own α map, so the
network predicts 8 × 3 = 24 maps.

**Why per-pixel α.** A global curve (like gamma) gives every pixel with the same value the same
output. Per-pixel α lets dark regions be lifted more than bright ones in the same image, which is
exactly the weakness of the classical baselines found in Milestone 1.

### The network (DCE-Net)

Seven 3×3 conv layers with 32 channels and ReLU. Layers 5, 6 and 7 take the concatenation of the
previous output with layers 3, 2 and 1 respectively ("symmetric concatenation", like a U-Net
without down-sampling). The last layer outputs 24 channels through tanh, giving α in [−1, 1].

- No pooling or batch normalization: the paper says they break relations between neighbouring
  pixels, and without them the network runs on any image size.
- 79,416 trainable parameters, matching the paper (checked by a test).
- Weights start from N(0, 0.02) as in the paper, so α ≈ 0 initially and the untrained network
  outputs roughly its input.

### The four losses: what each enforces

1. **Spatial consistency.** Average the image (over RGB) in 4×4 regions. For every region and its
   four neighbours, the contrast |Y_i − Y_j| in the output should equal |I_i − I_j| in the input.
   Keeps edges and local structure. *Without it:* lower contrast, washed-out regions.
2. **Exposure control.** Mean intensity of each 16×16 patch should be close to E = 0.6. This is the
   only term that pushes the image to get brighter. *Without it:* dark regions stay dark (the
   paper's ablation), because α = 0 satisfies the other three losses almost perfectly.
3. **Colour constancy.** Gray-World assumption: over the whole image, mean R, G and B should be
   equal. Penalizes squared differences of the channel means. *Without it:* strong colour casts,
   since each channel's curve can drift independently.
4. **Illumination smoothness.** Total variation of the α maps: neighbouring pixels should get
   similar curves. *Without it:* blotchy artifacts between regions.

Total: `L = L_spa + L_exp + 0.5·L_col + 20·L_tvA` (paper weights). All four weights are in
`config.yaml`, so an ablation is a config change (set a weight to 0).

### Why it trains without ground truth

None of the losses looks at the normal-light image. They compare the output with the input
(structure), with a fixed brightness target (exposure), with itself (colour balance), and
constrain the α maps (smoothness). Together they describe what a well-enhanced image looks like, so
any dark image is a valid training example. The `high/` images are used only to measure PSNR/SSIM.

Honest caveat: the best checkpoint is *selected* by validation PSNR, which uses the 35 validation
reference images. Training is zero-reference; model selection is not.

### Verification against the paper and the official code

Compared with the paper (arXiv 2001.06826v2) and the official repository `Li-Chongyi/Zero-DCE`
(`model.py`, `Myloss.py`, `lowlight_train.py`). Where they disagree I followed the paper, as the
project brief asks.

| Item | Paper | Official code | This project |
|---|---|---|---|
| Architecture | 7 conv, 32 ch, symmetric concat, tanh, 24 maps | Same (concat x3+x4, x2+x5, x1+x6) | Same; 79,416 params |
| Curve | x + α·x(1−x) | x + r·(x² − x), i.e. α = −r | Paper form (equivalent, tanh is symmetric) |
| Spatial loss | (\|Y_i−Y_j\| − \|I_i−I_j\|)², 4×4, 4 neighbours | Signed differences; zero padding adds border terms that penalize changing border brightness; two computed variables unused | Paper form, interior neighbours only |
| Exposure loss | mean \|Y_k − E\|, 16×16, E = 0.6 | mean (Y_k − E)² | Paper form |
| Colour loss | Σ (J_p − J_q)² | sqrt(Σ ((J_p − J_q)²)²) | Paper form |
| Smoothness loss | (\|∇x A\| + \|∇y A\|)² | \|∇x A\|² + \|∇y A\|² | Paper form, averaged over pixels |
| Loss weights | spa 1, exp 1, col 0.5, tvA 20 | spa 1, exp 10, col 5, tvA 200 | Paper (configurable) |
| Init | N(0, 0.02), constant bias | N(0, 0.02), default bias | N(0, 0.02), bias 0 |
| Optimizer | Adam (default params), lr 1e-4, batch 8 | Adam lr 1e-4, weight decay 1e-4, grad-norm clip 0.1 | Adam lr 1e-4, no weight decay, clip 0.1 |
| Training images | SICE Part1, 2,422 images resized to 512×512 | Resized to 256×256 | LOL low-light, random 256×256 crops + flips |

**What I could not verify:**
- The equations do not say how sums are normalized over pixels and batch. I average over pixels
  and batch everywhere. That sets the scale of each term and so how the weights balance them.
- The official code uses different formulas *and* different weights, so it is unclear whether the
  paper's weights were tuned for the formulas as printed. If the paper version trains badly
  (colour casts, no brightening), the first thing to try is the official formulation.
- The detailed architecture is in the supplementary material, which I did not read; the main text,
  the official code and the parameter count agree.
- The paper does not state the number of epochs or gradient clipping. Epochs = 200 (official
  script default); clipping at 0.1 is from the code.

### Difference from the paper: training data

The paper trains on SICE multi-exposure sequences (under- *and* over-exposed images). Here
Zero-DCE trains on the 450 LOL low-light training images only. Consequences:
- The model never sees over-exposed inputs, so it should not be expected to darken them.
- **Exposure target vs LOL references.** The LOL reference images have mean brightness 0.455
  (our485) and 0.451 (eval15), measured on this dataset; the low-light inputs average 0.060. The
  target is E = 0.6. The paper found little visual difference for E in [0.4, 0.7], but PSNR against
  LOL references rewards matching their brightness, so E = 0.6 will likely produce outputs brighter
  than the references and cost PSNR. Changing E would deviate from the paper, so the default stays.

### Training engineering

- **Resumable.** `last.pth` stores model, optimizer, epoch, best PSNR and history; rerunning the
  same command resumes. Checkpoints are written to a temp file and renamed, so a Colab disconnect
  during a save cannot corrupt the previous checkpoint.
- **Reproducible.** Seeds are fixed and the data RNG is re-seeded every epoch from (seed, epoch),
  so a resumed run sees exactly the same crops and order as an uninterrupted one (a test checks
  this on CPU). GPU kernels can still add tiny nondeterministic differences.
- **Fast data loading.** The 450 training images are decoded once and kept in memory as uint8.
- **Checkpoints.** `best.pth` and `zerodce.pth` by validation PSNR; `last.pth` for resuming.
  `zerodce.pth` holds only model weights (about 320 KB) and is the file that gets committed.
- **Colab.** The notebook keeps Colab's preinstalled CUDA PyTorch and installs the rest of
  `requirements.txt`.

### Local CPU check

Five epochs on my laptop CPU on real LOL data (`python -m src.train --epochs 5 --device cpu
--num-workers 0 --output-dir checkpoints/cpu_check`), about 2 minutes per epoch. This run only
checks that training works; it is not the final model.

| Epoch | Train loss | Val loss | Val PSNR (dB) | Val SSIM |
|---|---|---|---|---|
| 1 | 0.527 | 0.462 | 10.56 | 0.349 |
| 2 | 0.335 | 0.294 | 15.49 | 0.441 |
| 3 | 0.217 | 0.185 | 13.54 | 0.357 |
| 4 | 0.169 | 0.170 | 13.47 | 0.352 |
| 5 | 0.160 | 0.162 | 13.14 | 0.347 |

What it showed:
- **The total loss falls every epoch**, on train and validation, so the model and losses work.
- **The other three terms start near zero and rise.** At epoch 1 the output is almost the input
  (α ≈ 0), which trivially satisfies spatial consistency, colour constancy and smoothness. As the
  network learns to brighten (exposure loss 0.52 → 0.07), those terms become non-zero. That is the
  trade-off the weights balance.
- **Validation PSNR peaks at epoch 2 and then falls while the loss keeps improving.** Mean
  output brightness on the validation images was 0.402 for the epoch-2 checkpoint and 0.528 at
  epoch 5, against 0.458 for the references. The exposure loss keeps pushing towards E = 0.6, past
  the references' brightness, so PSNR drops. This is the mismatch predicted above, and it is why
  the best checkpoint is chosen by validation PSNR rather than by loss.
- Lower zero-reference loss does not mean higher PSNR: the training objective and the evaluation
  metric measure different things.

### Why it is fast enough for CPU inference

79K parameters and seven small convolutions (the paper reports 5.21 GFLOPs for a 256×256 input).
The curve is a few multiply-adds per pixel. No pooling or attention, so cost grows linearly with
the number of pixels. Measured CPU timings come in Milestone 3.

### Alternatives considered and rejected

- **Following the official code instead of the paper.** The brief asks for the paper; the code
  versions are documented above as a fallback.
- **Resizing training images instead of cropping.** Resizing 600×400 to 256×256 distorts the aspect
  ratio and changes noise statistics; random crops keep native resolution.
- **Supervised training on LOL pairs.** A different method (the optional U-Net stretch goal).
  Zero-DCE's point is not needing pairs.
- **Batch normalization.** Excluded by the paper; its statistics differ between 256×256 crops and
  full images, and it couples pixels across the batch.

### Likely interview questions

- **Why must α be in [−1, 1]?** That range keeps the output in [0, 1] and the curve monotonic; tanh
  produces exactly that range.
- **Why predict a curve instead of the image?** A curve guarantees valid output ranges, preserves
  intensity order and needs a far smaller network; image-to-image networks can hallucinate content.
- **What happens without the exposure loss?** Nothing pushes the output brighter; α = 0 satisfies
  the remaining losses.
- **Which loss stops colour casts, and when does it fail?** Colour constancy (Gray-World). It fails
  on scenes genuinely dominated by one colour, such as a red wall or a sunset.
- **Is it really zero-reference if you select checkpoints by PSNR?** Training is; selection uses 35
  references. Without any, you would take the last epoch or use a no-reference metric.
- **How did you verify the implementation?** Parameter count matches the paper (79,416); tests
  check the curve's range, fixed points and monotonicity and each loss on synthetic inputs; a short
  CPU run on LOL confirmed the loss decreases and validation PSNR rises.
- **What did you change from the paper and why?** Training data (LOL instead of SICE, as the
  project requires), 256×256 random crops instead of 512×512 resizing, and gradient clipping from
  the official code. Everything else follows the paper.
