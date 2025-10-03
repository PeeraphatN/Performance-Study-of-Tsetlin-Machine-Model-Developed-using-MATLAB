@echo off
setlocal enabledelayedexpansion

REM ==== Read arguments from command line ====
set "start=%1"
set "step=%2"
set "end=%3"

REM ==== Check if arguments are missing and validate step ====
if "%start%"=="" (
    echo Usage: run_normal_xor_dynamic.bat START STEP END
    echo Example: run_normal_xor_dynamic.bat 2 2 20
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

REM ==== Loop and run separate MATLAB instances ====
for /L %%C in (%start%,%step%,%end%) do (
    echo Running NormalXOR.m with --clauses %%C
    start "Clause %%C" cmd /k matlab -batch "NormalXOR('clauses',%%C)"
)

echo All processes launched.
