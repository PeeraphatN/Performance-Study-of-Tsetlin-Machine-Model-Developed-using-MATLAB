@echo off
setlocal enabledelayedexpansion

REM ==== Read arguments from command line ====
set "start=%1"
set "step=%2"
set "end=%3"

REM ==== Check if arguments are missing ====
if "%start%"=="" (
    echo Usage: run_mnist_dynamic.bat START STEP END
    echo Example: run_mnist_dynamic.bat 100 100 1000
    exit /b
)

REM ==== Loop and run separate python instances ====
for /L %%C in (%start%,%step%,%end%) do (
    echo Running MNISTDemo.py with --clause %%C
    start "Clause %%C" cmd /k python MNISTDemo.py --clause %%C
)

echo All processes launched.