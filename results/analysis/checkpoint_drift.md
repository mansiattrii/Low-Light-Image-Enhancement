# Checkpoint drift on the validation split (35 images)

Means over images. Corrected PSNR rescales the output to the reference's mean brightness (global) or per-channel means (colour); diagnostic only.

| Run | Checkpoint | Epoch | PSNR | Brightness-corrected PSNR | Colour-corrected PSNR | SSIM | Brightness (ref) | Colour-cast error | Contrast (ref) |
|---|---|---|---|---|---|---|---|---|---|
| zerodce | best | 2 | 15.49 | 18.51 | 18.92 | 0.441 | 0.403 (0.458) | 0.055 | 0.196 (0.180) |
| zerodce | last | 200 | 11.61 | 13.93 | 14.15 | 0.339 | 0.564 (0.458) | 0.095 | 0.143 (0.180) |
| zerodce_e045 | best | 3 | 14.65 | 15.48 | 15.73 | 0.380 | 0.382 (0.458) | 0.072 | 0.134 (0.180) |
| zerodce_e045 | last | 200 | 12.98 | 13.54 | 13.75 | 0.357 | 0.438 (0.458) | 0.082 | 0.122 (0.180) |
| zerodce_official | best | 14 | 16.08 | 18.67 | 19.01 | 0.427 | 0.433 (0.458) | 0.061 | 0.181 (0.180) |
| zerodce_official | last | 200 | 15.22 | 18.41 | 18.64 | 0.412 | 0.499 (0.458) | 0.069 | 0.190 (0.180) |
