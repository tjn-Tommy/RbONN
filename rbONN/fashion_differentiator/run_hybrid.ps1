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
    & $PY -m rbONN.fashion_differentiator.benchmark_hybrid  @args
}

# --- baseline at K=20 (default lr/batch/epochs) ----------------------------
Run-Opt --comparator optical    --name hyb_optical
Run-Opt --comparator optical    --name hyb_incoherent
Run-Opt --comparator optical    --name hyb_linear

Write-Host "`n=== sweep complete -- metrics in rbONN\fashion_differentiater\output\ ===" -ForegroundColor Green
