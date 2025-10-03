@echo off
setlocal enabledelayedexpansion

REM ==== Read arguments from command line ====
set "start=%1"
set "step=%2"
set "end=%3"

REM ==== Check if arguments are missing and validate step ====
if "%start%"=="" (
    echo Usage: run_noisy_dynamic.bat START STEP END
    echo Example: run_noisy_dynamic.bat 2 2 20
    exit /b
)

REM Ensure step is provided; default to 1 if empty
if "%step%"=="" (
    set "step=1"
)

REM Prevent infinite loop: step must not be zero
if "%step%"=="0" (
    echo ERROR: STEP must not be 0.
    exit /b 1
)

REM ==== Loop and run separate python instances ====
for /L %%C in (%start%,%step%,%end%) do (
    echo Running NoisyXORDemo.py with --clauses %%C
    start "Clause %%C" cmd /k python NoisyXORDemo.py --clauses %%C
)

echo All processes launched.
