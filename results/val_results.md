# Results on the validation split

- Images: 35 (first image 600x400)
- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over channels. Mean ± sample std over images.
- Brightness: mean output intensity in [0, 1]; the reference images average 0.458.
- Time: mean wall-clock time per image for the enhancement call only (no file I/O), after a warm-up call.
- Machine: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Windows, 8 PyTorch threads; GPU: NVIDIA GeForce RTX 3060 Laptop GPU. Python 3.11.9, PyTorch 2.14.1+cu132, OpenCV 5.0.0, scikit-image 0.26.0
- Generated: 2026-10-14 by `python -m src.evaluate`

| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | Brightness | CPU ms / image | GPU ms / image | Learned params |
|---|---|---|---|---|---|---|---|
| input (no enhancement) | - | 7.22 ± 1.96 | 0.134 ± 0.093 | 0.044 | 0.0 | – | 0 |
| gamma | gamma=0.25 | 16.39 ± 3.45 | 0.552 ± 0.149 | 0.391 | 7.5 | – | 0 |
| hist_eq | luma (YCrCb) | 15.28 ± 2.24 | 0.388 ± 0.135 | 0.506 | 4.6 | – | 0 |
| clahe | clip=32.0, tile=2 | 15.50 ± 2.32 | 0.386 ± 0.097 | 0.433 | 5.4 | – | 0 |
| zerodce | E=0.6 (paper) | 15.49 ± 4.43 | 0.441 ± 0.157 | 0.403 | 253.1 | 11.4 | 79,416 |
| zerodce_e045 | E=0.45 (ablation: mean brightness of training references) | 14.65 ± 2.44 | 0.380 ± 0.149 | 0.382 | 250.9 | 11.4 | 79,416 |
| zerodce_official | E=0.6, official-code loss formulas and weights | 16.08 ± 3.85 | 0.427 ± 0.154 | 0.433 | 250.5 | 11.3 | 79,416 |
