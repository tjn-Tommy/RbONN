# ---------------------------------------------------------------------------
# Tuning sweep: push the Fashion-MNIST same/different optical MLP (K=20 hidden
# |S|^2 units + linear readout) to its best accuracy.
#
# Run:   .\rbONN\fashion_differentiater\run_optical_sweep.ps1
# Watch: http://localhost:7860   (trackio project: rbONN_fashion_diff)
# Each run writes rbONN\fashion_differentiater\output\metrics_<name>.json
# Comment out (#) any line you don't want.
# ---------------------------------------------------------------------------
$PY   = "C:\Users\cusgadmin\.conda\envs\rbonn\python.exe"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repo

function Run-Opt {
    Write-Host "`n=== optical: $($args -join ' ') ===" -ForegroundColor Cyan
    & $PY -m rbONN.fashion_differentiater.benchmark_optical @args
}

# --- baseline at K=20 (default lr/batch/epochs) ----------------------------
Run-Opt --name f_h20_base   --hidden 20 --epochs 120

# --- learning-rate sweep ---------------------------------------------------
Run-Opt --name f_h20_lr3e-3 --hidden 20 --epochs 120 --lr 3e-3
Run-Opt --name f_h20_lr3e-2 --hidden 20 --epochs 120 --lr 3e-2

# --- batch-size sweep (smaller batch helped the digit task) ----------------
Run-Opt --name f_h20_bs128  --hidden 20 --epochs 120 --batch-size 128
Run-Opt --name f_h20_bs64   --hidden 20 --epochs 120 --batch-size 64

# --- longer schedule -------------------------------------------------------
Run-Opt --name f_h20_long   --hidden 20 --epochs 250

# --- best-guess combo: tuned lr + small batch + long schedule --------------
Run-Opt --name f_h20_best   --hidden 20 --epochs 250 --lr 3e-2 --batch-size 64

Write-Host "`n=== sweep complete -- metrics in rbONN\fashion_differentiater\output\ ===" -ForegroundColor Green
