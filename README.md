# Low-Light Image Enhancement

Brightens very dark photos and compares classical image processing (gamma correction, histogram
equalization, CLAHE) with Zero-DCE, a lightweight deep curve-estimation model implemented from its
paper. Quality is measured with PSNR and SSIM on the LOL dataset, and a denoising study checks
where removing noise helps.

> Work in progress. Milestone 1 (data and classical baselines) is complete.

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Download LOL (v1) manually from the
[official project page](https://daooshee.github.io/BMVC2018website/) and place it as:

```
data/LOL/our485/low/*.png   data/LOL/our485/high/*.png   (485 training pairs)
data/LOL/eval15/low/*.png   data/LOL/eval15/high/*.png   (15 test pairs)
```

35 training pairs are held out for validation; their filenames are listed in
`data/splits/val.txt`. Final results are reported only on eval15.

## Usage

```powershell
python -m src.classical                 # tune classical baselines on the validation split
python -m src.evaluate                  # evaluate on eval15 -> results/results.md
python -m src.evaluate --split val      # evaluate on the validation split
pytest
ruff check .; ruff format --check .
```

All settings live in `config.yaml`.

## Results

See [results/results.md](results/results.md). The full write-up will be added in the final
milestone.
