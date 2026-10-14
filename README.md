# Low-Light Image Enhancement

Brightens very dark photos and compares classical image processing (gamma correction, histogram
equalization, CLAHE) with Zero-DCE, a lightweight deep curve-estimation model implemented from its
paper. Quality is measured with PSNR and SSIM on the LOL dataset, and a denoising study checks
where removing noise helps.

> Work in progress. Milestones 1 (data and classical baselines) and 2 (Zero-DCE model and
> training code) are complete.

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

## Training Zero-DCE

Zero-DCE is trained with zero-reference losses on the LOL low-light images only; the normal-light
images are used for validation PSNR and evaluation. The paper trained on the SICE multi-exposure
dataset instead, so results are not directly comparable with the paper's numbers.

On Google Colab (GPU), open `notebooks/train_colab.ipynb`, set `REPO_URL`, upload `LOL.zip` to
`MyDrive/low-light-enhancement/`, and run all cells. If the session disconnects, run all cells
again: training resumes from the last checkpoint on Drive.

Locally (CPU, slow; for checking only):

```powershell
python -m src.train --epochs 2 --device cpu --num-workers 0 --output-dir checkpoints/cpu_check
```

## Results

See [results/results.md](results/results.md). The full write-up will be added in the final
milestone.
