@echo off
REM MURATURA - Launcher Windows
REM Avvia FreeCAD come Muratura Edition

echo ================================================
echo MURATURA - Analisi Strutturale Edifici Muratura
echo ================================================

set MURATURA_DIR=%~dp0
set FREECAD_DIR=%MURATURA_DIR%freecad
set WORKBENCH_DIR=%MURATURA_DIR%muratura\workbench

REM Copia workbench in FreeCAD Mod se non esiste
if not exist "%FREECAD_DIR%\Mod\Muratura" (
    echo Installazione workbench Muratura...
    xcopy "%WORKBENCH_DIR%" "%FREECAD_DIR%\Mod\Muratura" /E /I /Y /Q
)

REM Aggiungi path al PYTHONPATH
set PYTHONPATH=%MURATURA_DIR%muratura;%PYTHONPATH%

REM Avvia FreeCAD
echo Avvio Muratura...
start "" "%FREECAD_DIR%\bin\FreeCAD.exe"
