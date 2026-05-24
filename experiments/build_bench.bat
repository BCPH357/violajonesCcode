@echo off
REM Build bench_vj.exe via MSYS2 (output to MSYS2 home dir to allow execution)
REM Run from repo root: experiments\build_bench.bat

set MSYS2_BASH=C:\msys64\usr\bin\bash.exe

if not exist "%MSYS2_BASH%" (
    echo ERROR: MSYS2 not found at %MSYS2_BASH%
    echo Install MSYS2 from https://www.msys2.org/
    exit /b 1
)

REM Get repo root as MSYS2 path (assumes drive letter is one char)
set "REPO_WIN=%~dp0.."
set "REPO_WIN=%REPO_WIN:\=/%"
set "REPO_MSYS=/%REPO_WIN:~0,1%%REPO_WIN:~2%"

"%MSYS2_BASH%" -l -c "export PATH=/mingw64/bin:/usr/bin:$PATH && cd '%REPO_MSYS%' && gcc -O2 -o ~/bench_vj.exe experiments/bench_vj.c src/vj_integral.c src/vj_detect.c src/vj_cascade_data.c -lm && echo BUILD_OK bench_vj.exe"

REM Also rebuild test_vj.exe
"%MSYS2_BASH%" -l -c "export PATH=/mingw64/bin:/usr/bin:$PATH && cd '%REPO_MSYS%' && gcc -O2 -o ~/test_vj.exe src/vj_integral.c src/vj_detect.c src/vj_cascade_data.c test/test_main.c -lm && echo BUILD_OK test_vj.exe"
