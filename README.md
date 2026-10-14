# Low-Light Image Enhancement

Brightens very dark photos and compares classical image processing (gamma correction, histogram
equalization, CLAHE) with Zero-DCE, a lightweight deep curve-estimation model implemented from its
paper. Quality is measured with PSNR and SSIM on the LOL dataset, and a denoising study checks
where removing noise helps.

> Work in progress. Milestones 1 (data and classical baselines), 2 (Zero-DCE model and training
> code) and 3 (training and full evaluation) are complete.

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch, which is enough for inference. To train on
an NVIDIA GPU, replace it with the CUDA build of the same version (check `nvidia-smi` for the
CUDA version your driver supports):

```powershell
pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.cuda.is_available())"
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
python -m src.figures                   # comparison grid, failure cases, checkpoint comparison
python -m src.analysis                  # best-vs-last checkpoint diagnostic (needs checkpoints/)
python -m src.enhance --input dark.jpg --output out.png --method zerodce
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

The weights in `weights/` were trained locally on an RTX 3060:

```powershell
# weights/zerodce.pth (default): loss formulas and weights of the official implementation
python -m src.train --device cuda --num-workers 0 --output-dir checkpoints/zerodce_official
# weights/zerodce_paper.pth: the paper's printed loss equations and weights
python -m src.train --device cuda --num-workers 0 --loss-formulation paper --output-dir checkpoints/zerodce_e060
# weights/zerodce_e045.pth: paper equations with exposure level 0.45 (ablation)
python -m src.train --device cuda --num-workers 0 --loss-formulation paper --exposure-level 0.45 --output-dir checkpoints/zerodce_e045
```

The paper's equations as printed give a much weaker smoothness term than the official code, and
training with them degrades after a few epochs; see `NOTES.md` (Milestone 3).

## Results

See [results/results.md](results/results.md). The full write-up will be added in the final
milestone.
