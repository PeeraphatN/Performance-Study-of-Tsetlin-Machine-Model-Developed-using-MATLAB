@echo off
setlocal enabledelayedexpansion

REM ==== Read arguments from command line ====
set "start=%1"
set "step=%2"
set "end=%3"

REM ==== Check if arguments are missing ====
if "%start%"=="" (
    echo Usage: run_normal_xor_pure.bat START STEP END
    echo Example: run_normal_xor_pure.bat 10 10 100
    exit /b
)

REM ==== Loop and run separate python instances ====
for /L %%C in (%start%,%step%,%end%) do (
    echo Running NormalXORPure.py with --clauses %%C
    start "Clause %%C" cmd /k python NormalXORPure.py --clauses %%C
)

echo All processes launched.