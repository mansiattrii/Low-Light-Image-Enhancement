# Results on eval15 (test set): denoising study

- Images: 15 (first image 600x400)
- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over channels. Mean ± sample std over images.
- Brightness: mean output intensity in [0, 1]; the reference images average 0.451.
- Time: wall-clock time of the enhancement call only (no file I/O), fastest of 3 runs per image after a warm-up call, averaged over images.
- Machine: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Windows, 8 PyTorch threads; GPU: NVIDIA GeForce RTX 3060 Laptop GPU. Python 3.11.9, PyTorch 2.14.1+cu132, OpenCV 5.0.0, scikit-image 0.26.0
- Generated: 2026-10-18 by `python -m src.denoise`

| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | Brightness | CPU ms / image | GPU ms / image | Learned params |
|---|---|---|---|---|---|---|---|
| zerodce (no denoising) | - | 16.88 ± 3.55 | 0.525 ± 0.112 | 0.506 | 222.8 | – | 79,416 |
| zerodce->nlm | h=20 | 17.49 ± 3.77 | 0.691 ± 0.087 | 0.503 | 590.3 | – | 79,416 |
| nlm->zerodce | h=10 | 17.50 ± 3.92 | 0.696 ± 0.104 | 0.496 | 587.1 | – | 79,416 |
| zerodce->bilateral | sigma_color=1.0, sigma_space=2 | 17.82 ± 4.06 | 0.745 ± 0.090 | 0.506 | 223.5 | – | 79,416 |
| bilateral->zerodce | sigma_color=0.1, sigma_space=2 | 17.73 ± 4.08 | 0.755 ± 0.094 | 0.513 | 225.7 | – | 79,416 |

## Estimated noise in the output

Noise standard deviation in 8-bit levels, estimated from each saved output without a reference (median absolute Haar diagonal detail / 0.6745), mean ± sample std over images. Lower means smoother; it cannot tell noise from fine texture, so an output well below the reference has lost real detail.

| Output | Noise σ (8-bit levels) |
|---|---|
| reference | 1.88 ± 0.60 |
| zerodce (no denoising) | 9.88 ± 3.88 |
| zerodce->nlm | 0.48 ± 0.58 |
| nlm->zerodce | 0.01 ± 0.00 |
| zerodce->bilateral | 0.58 ± 0.26 |
| bilateral->zerodce | 0.59 ± 0.25 |
