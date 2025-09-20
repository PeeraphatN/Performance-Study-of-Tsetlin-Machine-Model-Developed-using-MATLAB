@echo off
setlocal enabledelayedexpansion

REM ==== Read arguments from command line ====
set "start=%1"
set "step=%2"
set "end=%3"

REM ==== Check if arguments are missing ====
if "%start%"=="" (
    echo Usage: run_noisy_pure.bat START STEP END
    echo Example: run_noisy_pure.bat 2 2 20
    exit /b
)

REM ==== Loop and run separate python instances ====
for /L %%C in (%start%,%step%,%end%) do (
    echo Running NoisyXORPure.py with --clauses %%C
    start "Clause %%C" cmd /k python NoisyXORPure.py --clauses %%C
)

echo All processes launched.
