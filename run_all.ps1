$ErrorActionPreference = "Stop"
& py -m ensurepip --upgrade
& py -m pip install --upgrade pip

if (-not (Test-Path ".\.venv\Scripts\python.exe")) { & py -3 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m ensurepip --upgrade
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

& $py scripts/train_supervised.py
& $py scripts/unsupervised_kmeans.py
& $py scripts/analyze_and_report.py
Write-Host "Done."
