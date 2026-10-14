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
  the best checkpoint is chosen by validation PSNR rather than by loss. *(Milestone 3 showed
  brightness is not the main cause: the drop is lost contrast and haloes caused by a weak
  smoothness term. See "Why validation PSNR fell" below.)*
- Lower zero-reference loss does not mean higher PSNR: the training objective and the evaluation
  metric measure different things.

### Why it is fast enough for CPU inference

79K parameters and seven small convolutions (the paper reports 5.21 GFLOPs for a 256×256 input).
The curve is a few multiply-adds per pixel. No pooling or attention, so cost grows linearly with
the number of pixels. Measured timings: see Milestone 3.

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

---

## Milestone 3: training and full evaluation

### What was built

- `src/enhance.py`: the single inference function used by the CLI, evaluation and later the API
  and demo. Handles grayscale, RGBA, uint8 or float input, odd sizes, and caps the longer side at
  `inference.max_side` (2048).
- `src/evaluate.py` now evaluates Zero-DCE variants as well and reports mean output brightness and
  GPU time (when a GPU exists) next to PSNR, SSIM, CPU time and parameter count.
- `src/figures.py`: comparison grid, failure cases, best-vs-last checkpoint figure, own-photos
  grid (when `data/own/` has images) and copies of the training logs.
- `src/analysis.py`: the checkpoint drift diagnostic explained below.
- `src/losses.py`: the official code's loss formulas as a second formulation
  (`loss_formulation: official`), checked against a transcription of the official `Myloss.py`.

### How training was run

- On my laptop's RTX 3060 (6 GB), not Colab. The first attempt ran on the CPU because the
  installed PyTorch was the CPU-only build (about 4 minutes per epoch with two runs in parallel).
  Installing the CUDA build of the same version brought it to 10–18 seconds per epoch. The Colab
  notebook still works and is documented as the alternative.
- Three runs, 200 epochs each, everything identical except the loss:

| Run | Loss | Best epoch (val PSNR) | Training time |
|---|---|---|---|
| `zerodce` | paper equations, paper weights, E = 0.6 | 2 (15.49 dB) | 58 min (two runs sharing the GPU) |
| `zerodce_e045` | paper equations, E = 0.45 (ablation) | 3 (14.65 dB) | 58 min (shared) |
| `zerodce_official` | official-code formulas and weights, E = 0.6 | 14 (16.08 dB) | 44 min (alone; one restart from epoch 92) |

The official run was stopped once at epoch 92 because the laptop ran low on memory, and resumed
from `last.pth`. The resume logic made this a non-event.

E = 0.45 was set from the mean brightness of the 450 training-split references (0.455). That uses
reference images to choose a hyperparameter, so this variant is not strictly zero-reference.

### Results on eval15 (`results/results.md`)

| Method | PSNR (dB) | SSIM | Brightness (ref 0.451) | CPU ms | GPU ms |
|---|---|---|---|---|---|
| Input | 7.77 ± 2.62 | 0.195 | 0.059 | – | – |
| Gamma 0.25 | 16.87 ± 2.83 | **0.634** | 0.437 | 7.1 | – |
| Histogram equalization | 14.24 ± 3.14 | 0.419 | 0.512 | 4.2 | – |
| CLAHE | 14.20 ± 1.77 | 0.392 | 0.467 | 5.3 | – |
| Zero-DCE, paper losses | 16.64 ± 3.59 | 0.544 | 0.492 | 250.6 | 11.6 |
| Zero-DCE, paper losses, E = 0.45 | 14.47 ± 2.08 | 0.469 | 0.416 | 251.3 | 11.8 |
| Zero-DCE, official losses | **16.88 ± 3.55** | 0.525 | 0.506 | 253.0 | 11.8 |

What this says:
- **Zero-DCE ties tuned gamma on PSNR and loses on SSIM.** The 0.01 dB gap between the official
  variant and gamma is meaningless next to standard deviations of about 3 dB on 15 images.
- **Gamma had an advantage Zero-DCE did not.** Its one parameter was tuned on validation PSNR
  against references. Zero-DCE never sees references in training; only checkpoint selection does.
- **Zero-DCE spreads its errors differently.** Its standard deviation is larger: it wins clearly on
  some images (e.g. 111.png, 20.91 dB vs 18.21 for gamma) and loses badly on others (179.png).
- **Lowering E to 0.45 hurt.** Matching average brightness does not help if the model produces flat,
  hazy images (below).

### Why validation PSNR fell during training (`results/analysis/checkpoint_drift.md`)

With the paper's equations, validation PSNR peaked at epoch 2–3 and then fell for the remaining
~200 epochs while the training loss kept improving. To find out why, `src/analysis.py` scores the
best and last checkpoints three ways: plain PSNR, PSNR after rescaling the output to the
reference's mean brightness, and PSNR after rescaling each colour channel to the reference's
channel means (diagnostic only; these use the references).

| Run | Checkpoint | PSNR | Brightness-corrected | Colour-corrected | Contrast (ref 0.180) |
|---|---|---|---|---|---|
| paper | epoch 2 | 15.49 | 18.51 | 18.92 | 0.196 |
| paper | epoch 200 | 11.61 | 13.93 | 14.15 | 0.143 |
| official | epoch 14 | 16.08 | 18.67 | 19.01 | 0.181 |
| official | epoch 200 | 15.22 | 18.41 | 18.64 | 0.190 |

- Brightness is not the explanation. Removing the brightness error helps both checkpoints, and
  afterwards the gap between them is still 4.6 dB (18.51 vs 13.93). Removing the colour error
  as well changes almost nothing (about 0.2–0.4 dB).
- What is left is structure: contrast (std of luma) fell from 0.196 to 0.143. The epoch-200 images in
  `figures/checkpoint_comparison.png` show a grey-pink haze, embossed halos around edges and
  amplified noise: the classic look of curve maps that change too sharply from pixel to pixel.
- **Root cause: the illumination smoothness term is far weaker in the paper's equations as I
  implemented them than in the official code.** Relative to the exposure term, the official code's
  smoothness penalty (`200 · 2 · Σ mean(∇A²)`) is roughly 80–160 times stronger than the paper's
  printed version with weight 20 averaged over 8 iterations. This is exactly the normalization
  ambiguity listed under "could not verify" in Milestone 2.
- **The official formulation fixes it.** Its last checkpoint keeps the reference's contrast, and
  its corrected PSNR barely changes between epoch 14 and 200; validation PSNR stays in a
  15–16 dB band for the whole run instead of collapsing.

**Lesson:** "following the paper" was not enough. A paper's equations can leave out scale factors
that decide whether training works; the reference code is part of the specification. Comparing
the two was only possible because both are implemented behind one config switch with tests that
check the official version against a transcription of the original code.

### Failure cases (`results/failure_cases.md`)

- Gamma fails on the darkest input (55.png): one exponent cannot brighten it enough, and it lifts
  shadow noise.
- Histogram equalization and CLAHE fail on 179.png, whose reference is genuinely dark: they force
  the full intensity range and nearly double the brightness. They also wash out colour, because
  only luma is changed and the chroma of a dark image is small.
- Zero-DCE fails on the same 179.png for its own reasons: the exposure loss pushes every patch to
  0.6, and the Gray-World colour loss turns a brown wall pink. These are assumptions built into the
  losses, not training bugs.
- Every method amplifies noise in the darkest regions, which motivates Milestone 4.

### Decision after review: the official formulation is the default

After Milestone 3 the official-losses model became the project's default Zero-DCE:

- `weights/zerodce.pth` is now the official-losses model (best epoch 14). The paper-equations
  model moved to `weights/zerodce_paper.pth`; the E = 0.45 ablation stays in
  `weights/zerodce_e045.pth`.
- `config.yaml` now has `loss_formulation: official`, so `python -m src.train` reproduces the
  default model. The paper's weights stay in the config and `--loss-formulation paper` trains the
  paper version.
- In `results/` the methods are now named `zerodce` (official losses), `zerodce_paper` and
  `zerodce_paper_e045`. The tables in the Milestone 3 section above use the earlier names; the
  numbers are the same.

Why: both score about the same on eval15 (16.88 vs 16.64 dB PSNR, 0.525 vs 0.544 SSIM), but only
the official formulation trains stably. The paper version's kept checkpoint is an epoch-2
snapshot taken just before training degraded it. The API, demo and denoising study will see
photos unlike LOL, and a stably trained model is the safer choice there.

This deviates from the original brief (loss weights defaulting to the paper's values). The paper
version is still implemented, tested, trained and reported.

### Speed (eval15, 600×400, `results/results.md`)

- Zero-DCE: 250–430 ms per image on the laptop CPU (8 PyTorch threads) across the evaluation
  runs, and 12–16 ms on the RTX 3060 including copying the image to and from the GPU.
- Classical methods: 4–11 ms on the CPU. Zero-DCE is roughly 40× slower on the CPU but still fast
  enough for an interactive demo.
- **Timings on a laptop are noisy.** The same architecture measured 294 ms and 384 ms in one run,
  and every method slowed down when Discord and Chrome were busy. The evaluation therefore times
  each image three times and keeps the fastest run (standard benchmarking practice: slower runs
  mostly measure interference), but differences under ~30% between runs on this machine are not
  meaningful. Quality metrics are unaffected and identical across runs.

### Alternatives considered and rejected

- **Reporting the last checkpoint.** It is what the loss converged to, but with the paper's
  equations it is visibly broken; choosing by validation PSNR is standard and stated openly.
- **Silently switching to the official losses.** The brief asks for the paper. Both are kept,
  both are reported, and the difference is explained.
- **Tuning E on eval15.** Never. E = 0.45 came from training-split references.
- **Early stopping on the zero-reference loss.** The loss kept falling while quality fell, so it is
  the wrong signal for stopping.

### Likely interview questions

- **Does Zero-DCE beat the classical baseline?** On PSNR it ties tuned gamma (16.88 vs 16.87 dB);
  on SSIM gamma is better (0.634 vs 0.525). Gamma was tuned with references; Zero-DCE was not.
  Zero-DCE adapts per image and per pixel, which shows on some images but not on average.
- **Why did your model's PSNR peak at epoch 2?** With the paper's equations as printed, the
  smoothness penalty was too weak, so longer training produced sharp, haloed curve maps and lost
  contrast. The diagnostic (brightness- and colour-corrected PSNR) showed the gap stays after
  removing brightness and colour errors. The official code's formulation trains stably.
- **How did you diagnose it?** Separated the error into global brightness, colour balance and
  everything else by rescaling outputs to the reference's means, then looked at the images, then
  compared the loss scales with the official implementation.
- **Why choose checkpoints by PSNR if training is zero-reference?** Because the training loss does
  not track quality here. It needs a small labelled validation set, which is stated as a caveat.
- **Why is a lower exposure target worse even though it matches the references' brightness?**
  Brightness was not the main problem; flat, hazy outputs were.
- **CPU vs GPU timing?** 250–430 ms vs 12–16 ms per 600×400 image on my laptop; the GPU time
  includes transfers. Timing on a shared laptop is noisy, so I report a range.
- **What would you try next?** A stronger smoothness weight with the paper's formulas (to separate
  formula from weight effects), a no-reference quality metric for checkpoint selection, and
  Zero-DCE++ (downsampled curve estimation) for speed.
