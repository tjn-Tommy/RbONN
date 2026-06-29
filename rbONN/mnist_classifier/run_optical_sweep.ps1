# ---------------------------------------------------------------------------
# Tuning sweep for the single optical 784x10 |S|^2 layer on MNIST (PowerShell).
# Goal: push this fixed architecture to its best accuracy (linear baseline=92.68%).
#
# Run:   powershell -ExecutionPolicy Bypass -File rbONN\mnist\run_optical_sweep.ps1
#   or just:  .\rbONN\mnist\run_optical_sweep.ps1
# Watch: http://localhost:7860   (trackio project: rbONN_mnist_opt)
# Each run writes rbONN\mnist\output\metrics_<name>.json
# Comment out (#) any line you don't want.
# ---------------------------------------------------------------------------
$PY   = "C:\Users\cusgadmin\.conda\envs\rbonn\python.exe"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repo

function Run-Opt {
    Write-Host "`n=== optical: $($args -join ' ') ===" -ForegroundColor Cyan
    & $PY -m rbONN.mnist.benchmark_optical @args
}

# --- baseline + the 4x4-encoding sanity check (should be ~identical) --------
# Run-Opt --name opt_flat_base   --encoding flat  --epochs 120
# Run-Opt --name opt_patch4x4    --encoding patch --epochs 120    # expect == opt_flat_base

# # --- learning-rate / schedule / batch tuning -------------------------------
# Run-Opt --name opt_long        --encoding flat  --epochs 250
# Run-Opt --name opt_lr3e-3      --encoding flat  --epochs 120 --lr 3e-3
# Run-Opt --name opt_lr3e-2      --encoding flat  --epochs 120 --lr 3e-2
# Run-Opt --name opt_bs128       --encoding flat  --epochs 120 --batch-size 128

# # --- physics-motivated additions (the real gap-closers) --------------------
# Run-Opt --name opt_bias        --encoding flat  --epochs 120 --bias
# Run-Opt --name opt_scale       --encoding flat  --epochs 120 --logit-scale
# Run-Opt --name opt_bias_scale  --encoding flat  --epochs 150 --bias --logit-scale

# ===========================================================================
# PUSH FOR +1% : stack the sweep winners (bs128 best=91.99%) + untested knobs.
# Best single bet = bs64/128 + --bias.  Run these and keep the highest.
# ===========================================================================
Run-Opt --name opt_push_bias    --encoding flat --batch-size 128 --lr 0.03 --epochs 200 --bias --logit-scale
Run-Opt --name opt_push_bs64    --encoding flat --batch-size 64  --lr 0.03 --epochs 200 --bias --logit-scale
Run-Opt --name opt_push_adamw   --encoding flat --batch-size 128 --lr 0.02 --epochs 200 --bias --logit-scale --optimizer adamw --weight-decay 1e-4 --warmup 5 --label-smoothing 0.05

Write-Host "`n=== sweep complete -- metrics in rbONN\mnist\output\ ===" -ForegroundColor Green
