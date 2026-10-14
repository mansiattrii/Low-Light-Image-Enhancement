# Results on eval15 (test set)

- Images: 15 (first image 600x400)
- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over channels. Mean ± sample std over images.
- Brightness: mean output intensity in [0, 1]; the reference images average 0.451.
- Time: mean wall-clock time per image for the enhancement call only (no file I/O), after a warm-up call.
- Machine: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Windows, 8 PyTorch threads; GPU: NVIDIA GeForce RTX 3060 Laptop GPU. Python 3.11.9, PyTorch 2.14.1+cu132, OpenCV 5.0.0, scikit-image 0.26.0
- Generated: 2026-10-14 by `python -m src.evaluate`

| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | Brightness | CPU ms / image | GPU ms / image | Learned params |
|---|---|---|---|---|---|---|---|
| input (no enhancement) | - | 7.77 ± 2.62 | 0.195 ± 0.096 | 0.059 | 0.0 | – | 0 |
| gamma | gamma=0.25 | 16.87 ± 2.83 | 0.634 ± 0.103 | 0.437 | 7.1 | – | 0 |
| hist_eq | luma (YCrCb) | 14.24 ± 3.14 | 0.419 ± 0.101 | 0.512 | 4.2 | – | 0 |
| clahe | clip=32.0, tile=2 | 14.20 ± 1.77 | 0.392 ± 0.065 | 0.467 | 5.3 | – | 0 |
| zerodce | E=0.6 (paper) | 16.64 ± 3.59 | 0.544 ± 0.117 | 0.492 | 250.6 | 11.6 | 79,416 |
| zerodce_e045 | E=0.45 (ablation: mean brightness of training references) | 14.47 ± 2.08 | 0.469 ± 0.093 | 0.416 | 251.3 | 11.8 | 79,416 |
| zerodce_official | E=0.6, official-code loss formulas and weights | 16.88 ± 3.55 | 0.525 ± 0.112 | 0.506 | 253.0 | 11.8 | 79,416 |
