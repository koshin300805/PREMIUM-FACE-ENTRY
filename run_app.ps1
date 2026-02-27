# Run the app using the workspace virtualenv (avoids PowerShell quoting issues)
$python = Join-Path $PSScriptRoot 'venv312\Scripts\python.exe'
$app = Join-Path $PSScriptRoot 'app.py'

if (-Not (Test-Path $python)) {
    Write-Error "Python executable not found at $python. Activate your virtualenv or adjust the path."
    exit 1
}

& $python $app
