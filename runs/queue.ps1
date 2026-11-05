<#
    Run the bench queue: 4-bit ladder, fMNIST pairs, MNIST rerun, ladder again.

        powershell -ExecutionPolicy Bypass -File runs\queue.ps1 -DryRun   # rehearse, seconds
        powershell -ExecutionPolicy Bypass -File runs\queue.ps1           # the real ~36 h

    Shortest job first, so results bank early and the 19-hour one -- the one
    most likely to be interrupted -- goes last.  The ladder runs at BOTH ends:
    it does not anchor its own gain the way onn_infer does, so the pair of them
    is what says whether the bench held still across the whole queue.  Skip the
    second with -NoBracket.

    The GUI must not be holding the DAQ or the SLM.  Every job opens the DAQ
    first, so a busy machine stops that job before the panel is touched.

    Each job's output goes to its own log beside its CSV; this window gets the
    queue banners only.  Follow a running job from another window with

        Get-Content -Wait runs\mnist\mnist-rerun77_<stamp>.log

    A failing job does NOT stop the queue: these are independent measurements,
    and a wedged instrument on one is no reason to skip the next.  Exit codes
    are collected and printed in the summary.

    Nothing here scores anything.  Run the report.py beside each CSV afterwards;
    they are offline and cost seconds.
#>
param(
    [string]$Python = "C:\Users\weishan\.conda\envs\rbonn\python.exe",
    [string]$Calib  = "src\calib_data\1027_calib",
    [switch]$DryRun,
    [switch]$NoBracket
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot          # runs\ -> the repo root
Set-Location $repo
$stamp = Get-Date -Format "MMdd_HHmm"

$ladder = @("src\drafts\bit_acc_chara.py", "--step7", $Calib,
            "--out", "runs\4bit_acc_char")

$jobs = @(
    @{ Name = "4bit-before";    Dir = "runs\4bit_acc_char"; Est = "0.8 h"
       Args = $ladder }
    @{ Name = "fmnist-200pairs"; Dir = "runs\fmnist";       Est = "4.9 h"
       Args = @("src\onn_module\onn_infer.py",
                "--npz",   "runs\fmnist\bench_reference_200pairs.npz",
                "--calib", $Calib,
                "--out",   "runs\fmnist") }
    @{ Name = "mnist-rerun77";  Dir = "runs\mnist";         Est = "18.8 h"
       Args = @("src\onn_module\onn_infer.py",
                "--npz",   "runs\mnist\bench_reference_rerun77img.npz",
                "--calib", $Calib,
                "--out",   "runs\mnist") }
)
if (-not $NoBracket) {
    $jobs += @{ Name = "4bit-after"; Dir = "runs\4bit_acc_char"; Est = "0.8 h"
                Args = $ladder }
}

# ----------------------------------------------------------------------
# pre-flight -- a typo should cost a second, not a place in a 36 h queue
# ----------------------------------------------------------------------

Write-Output ""
Write-Output "queue $stamp   $($jobs.Count) jobs   calib $Calib"
if ($DryRun) {
    Write-Output "DRY RUN -- every job gets --dry-run; no instrument is opened"
}

$missing = @()
if (-not (Test-Path -LiteralPath $Python)) { $missing += $Python }
if (-not (Test-Path -LiteralPath $Calib))  { $missing += $Calib }
foreach ($j in $jobs) {
    if (-not (Test-Path -LiteralPath $j.Dir)) { $missing += $j.Dir }
    foreach ($a in $j.Args) {
        if (($a -like "*.py") -or ($a -like "*.npz")) {
            if (-not (Test-Path -LiteralPath $a)) { $missing += $a }
        }
    }
}
if ($missing.Count -gt 0) {
    Write-Output ""
    Write-Output "missing, nothing was run:"
    $missing | Sort-Object -Unique | ForEach-Object { Write-Output "   $_" }
    exit 1
}

$drive = $repo.Substring(0, 1)
$free = (Get-PSDrive -Name $drive).Free / 1GB
Write-Output ("free on {0}: {1:N0} GB" -f $drive, $free)
if ($free -lt 5) {
    Write-Output "under 5 GB free -- stopping before anything is written"
    exit 1
}

# ----------------------------------------------------------------------
# the queue
# ----------------------------------------------------------------------

$results = @()
$queueStart = Get-Date

foreach ($j in $jobs) {
    $n = $results.Count + 1
    $log = Join-Path $j.Dir "$($j.Name)_$stamp.log"
    $argv = $j.Args
    if ($DryRun) { $argv = $argv + "--dry-run" }

    Write-Output ""
    Write-Output ("=" * 72)
    Write-Output ("[{0}/{1}] {2}   est {3}   started {4:yyyy-MM-dd HH:mm:ss}" -f `
                  $n, $jobs.Count, $j.Name, $j.Est, (Get-Date))
    Write-Output "        log  $log"
    Write-Output ("=" * 72)

    $start = Get-Date
    # cmd does the stream merge, not PowerShell: 5.1 wraps a native command's
    # redirected stderr in an ErrorRecord and flips $? even on a clean exit,
    # which would turn an ordinary progress line into a failed job.
    $quoted = ($argv | ForEach-Object { '"' + $_ + '"' }) -join ' '
    cmd /c ('"' + $Python + '" ' + $quoted + ' > "' + $log + '" 2>&1')
    $code = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    $results += [pscustomobject]@{
        Job     = $j.Name
        Exit    = $code
        Elapsed = ("{0:hh\:mm\:ss}" -f $elapsed)
        Log     = $log
    }
    Write-Output ("        exit {0} after {1:hh\:mm\:ss}" -f $code, $elapsed)
    if ($code -ne 0) {
        Write-Output "        FAILED -- see the log; continuing to the next job"
    }
}

# ----------------------------------------------------------------------
# the summary
# ----------------------------------------------------------------------

$total = (Get-Date) - $queueStart
Write-Output ""
Write-Output ("=" * 72)
Write-Output ("queue finished in {0:hh\:mm\:ss}" -f $total)
Write-Output ""
$results | Format-Table -AutoSize | Out-String | Write-Output

Write-Output "written by this queue:"
$dirs = $jobs | ForEach-Object { $_.Dir } | Sort-Object -Unique
foreach ($d in $dirs) {
    Get-ChildItem -LiteralPath $d -Filter *.csv |
        Where-Object { $_.LastWriteTime -gt $queueStart } |
        ForEach-Object { Write-Output ("   {0}" -f $_.FullName) }
}

$bad = @($results | Where-Object { $_.Exit -ne 0 })
Write-Output ""
if ($bad.Count -gt 0) {
    Write-Output "$($bad.Count) of $($results.Count) job(s) failed."
    exit 1
}
Write-Output "all $($results.Count) jobs exited 0."
exit 0
