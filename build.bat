@echo off
REM Build the Viola-Jones test program (Windows, GCC/MinGW)
REM Usage: build.bat

gcc -Wall -Wextra -O2 ^
    src/vj_integral.c ^
    src/vj_detect.c ^
    src/vj_cascade_data.c ^
    test/test_main.c ^
    -lm -o test/test_vj.exe

if %ERRORLEVEL% EQU 0 (
    echo Build succeeded: test/test_vj.exe
) else (
    echo Build failed.
)
