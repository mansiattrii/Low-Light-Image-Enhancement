# Results on the validation split

- Images: 35 (first image 600x400)
- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over channels. Mean ± sample std over images.
- CPU time: mean wall-clock time per image for the enhancement call only (no file I/O).
- Machine: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Windows; Python 3.11.9, OpenCV 5.0.0, scikit-image 0.26.0
- Generated: 2026-10-14 by `python -m src.evaluate`

| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | CPU ms / image | Learned params |
|---|---|---|---|---|---|
| input (no enhancement) | - | 7.22 ± 1.96 | 0.134 ± 0.093 | 0.0 | 0 |
| gamma | gamma=0.25 | 16.39 ± 3.45 | 0.552 ± 0.149 | 8.2 | 0 |
| hist_eq | luma (YCrCb) | 15.28 ± 2.24 | 0.388 ± 0.135 | 6.3 | 0 |
| clahe | clip=32.0, tile=2 | 15.50 ± 2.32 | 0.386 ± 0.097 | 5.7 | 0 |
