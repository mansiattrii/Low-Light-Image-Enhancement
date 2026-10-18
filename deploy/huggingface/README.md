---
title: Low-Light Image Enhancement
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
short_description: Brighten dark photos with Zero-DCE or classical methods
---

# Low-light image enhancement

Upload a dark photo and compare Zero-DCE, a 79k-parameter network that predicts a brightening
curve for every pixel, with gamma correction, histogram equalization and CLAHE. An optional
bilateral filter removes the noise that brightening amplifies.

Runs on CPU; Zero-DCE takes a few hundred milliseconds per 600×400 image. Large photos are
downscaled so the longer side is at most 2048 pixels.
