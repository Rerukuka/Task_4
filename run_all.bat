@echo off
setlocal
py -m ensurepip --upgrade
py -m pip install --upgrade pip
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
set "PYEXE=.venv\Scripts\python.exe"
"%PYEXE%" -m ensurepip --upgrade
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r requirements.txt

"%PYEXE%" scripts\train_supervised.py || goto :fail
"%PYEXE%" scripts\unsupervised_kmeans.py || goto :fail
"%PYEXE%" scripts\analyze_and_report.py || goto :fail

echo Done.
pause
exit /b 0
:fail
echo [ERROR] See messages above.
pause
exit /b 1
