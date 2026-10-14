# Results on eval15 (test set)

- Images: 15 (first image 600x400)
- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over channels. Mean ± sample std over images.
- CPU time: mean wall-clock time per image for the enhancement call only (no file I/O).
- Machine: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Windows; Python 3.11.9, OpenCV 5.0.0, scikit-image 0.26.0
- Generated: 2026-10-14 by `python -m src.evaluate`

| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | CPU ms / image | Learned params |
|---|---|---|---|---|---|
| input (no enhancement) | - | 7.77 ± 2.62 | 0.195 ± 0.096 | 0.0 | 0 |
| gamma | gamma=0.25 | 16.87 ± 2.83 | 0.634 ± 0.103 | 7.2 | 0 |
| hist_eq | luma (YCrCb) | 14.24 ± 3.14 | 0.419 ± 0.101 | 4.3 | 0 |
| clahe | clip=32.0, tile=2 | 14.20 ± 1.77 | 0.392 ± 0.065 | 5.3 | 0 |
