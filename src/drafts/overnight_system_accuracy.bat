@echo off
rem Overnight system accuracy: RUNS runs, one started every hour.
rem Keeps each run's CSV and report PNG in OUT; the pattern JSON is deleted.
rem --pmt logs the PMT thermistor (Keithley 2400, ohms + TH10K degrees) next to
rem every read, so the night's drift can be read against detector temperature.
rem One GPIB, one 2400: do NOT run daq_pmt_monitor.py alongside this.
setlocal EnableDelayedExpansion
cd /d "C:\Users\weishan\Desktop\RbONN"

set "PY=C:\Users\weishan\.conda\envs\rbonn\python.exe"
set "STEP7=.\src\calib_data\calib_0917_1617\calib_step7_result_0917_1831.json"
rem OUT is stamped with THIS batch's start time, so a night of accuracy
rem runs can never land in the calibration folder it was built from.
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format MMdd_HHmm"') do set "OUT=.\src\calib_data\accuracy_%%d"
set RUNS=18
set PERIOD_S=3600

if not exist "%OUT%" mkdir "%OUT%"
for /f %%t in ('powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"') do set T0=%%t

for /l %%i in (1,1,%RUNS%) do (
    echo.
    echo ===== run %%i / %RUNS%  started !date! !time! =====
    "%PY%" .\src\calibration_module\steps\system_accuracy.py --step7 "%STEP7%" --out "%OUT%" --pmt
    if errorlevel 1 echo ***** run %%i exited with an error *****
    del /q "%OUT%\system_accuracy_*.json" 2>nul

    if %%i lss %RUNS% (
        powershell -NoProfile -Command "$w = !T0! + %%i * %PERIOD_S% - [int][double]::Parse((Get-Date -UFormat %%s)); if ($w -gt 0) { Write-Host ('waiting ' + $w + ' s for the next run'); Start-Sleep -Seconds $w }"
    )
)

echo.
echo ===== all %RUNS% runs done !date! !time! =====
endlocal
