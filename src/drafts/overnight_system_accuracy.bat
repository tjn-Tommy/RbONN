@echo off
rem Overnight system accuracy: 12 runs, one started every hour.
rem Keeps each run's CSV and report PNG in OUT; the pattern JSON is deleted.
setlocal EnableDelayedExpansion
cd /d "C:\Users\weishan\Desktop\RbONN"

set "PY=C:\Users\weishan\.conda\envs\rbonn\python.exe"
set "STEP7=.\src\calib_data\run_0916_10pairs_w15g35\calib_step7_result_0916_merged_ref2.json"
set "OUT=.\src\calib_data\run_0917_overnight"
set RUNS=12
set PERIOD_S=3600

if not exist "%OUT%" mkdir "%OUT%"
for /f %%t in ('powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"') do set T0=%%t

for /l %%i in (1,1,%RUNS%) do (
    echo.
    echo ===== run %%i / %RUNS%  started !date! !time! =====
    "%PY%" .\src\calibration_module\steps\system_accuracy.py --step7 "%STEP7%" --out "%OUT%"
    if errorlevel 1 echo ***** run %%i exited with an error *****
    del /q "%OUT%\system_accuracy_*.json" 2>nul

    if %%i lss %RUNS% (
        powershell -NoProfile -Command "$w = !T0! + %%i * %PERIOD_S% - [int][double]::Parse((Get-Date -UFormat %%s)); if ($w -gt 0) { Write-Host ('waiting ' + $w + ' s for the next run'); Start-Sleep -Seconds $w }"
    )
)

echo.
echo ===== all %RUNS% runs done !date! !time! =====
endlocal
