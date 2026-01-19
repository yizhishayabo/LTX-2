# Auto Captioning Script for LTX-2
# Usage: ./auto_caption.ps1 [API_KEY]

param (
    [string]$ApiKey
)

# Set directories
$ProjectRoot = "c:\Users\35401\project\ai-Training\LTX-2"
$InputIds = "c:\Users\35401\project\ai-Training\ai-Ds\completefile"
$ScriptPath = Join-Path $ProjectRoot "packages\ltx-trainer\scripts\caption_videos.py"

# Check if Python script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Python script not found at $ScriptPath"
    exit 1
}

# Check if Input directory exists
if (-not (Test-Path $InputIds)) {
    Write-Error "Input directory not found at $InputIds"
    exit 1
}

# Handle API Key
if (-not $ApiKey) {
    if ($env:GOOGLE_API_KEY) {
        $ApiKey = $env:GOOGLE_API_KEY
    }
    elseif ($env:GEMINI_API_KEY) {
        $ApiKey = $env:GEMINI_API_KEY
    }
    else {
        $ApiKey = Read-Host "Please enter your Gemini Flash API Key"
    }
}

if (-not $ApiKey) {
    Write-Error "API Key is required to run Gemini Flash captioning."
    exit 1
}

# Set environment variable for the python script
$env:GEMINI_API_KEY = $ApiKey

# Run the captioning script
Write-Host "Starting captioning process for videos in $InputIds..."
# Resolve Python executable in .venv
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RestScript = Join-Path $ProjectRoot "rest_caption.py"

if (-not (Test-Path $PythonExe)) {
    Write-Warning "Virtual environment python not found at $PythonExe. Falling back to system python."
    $PythonExe = "python"
}

# Run the captioning script
Write-Host "Starting captioning using REST API script..."
& $PythonExe $RestScript $InputIds $ApiKey

if ($LASTEXITCODE -eq 0) {
    Write-Host "Captioning completed successfully!" -ForegroundColor Green
}
else {
    Write-Error "Captioning failed with exit code $LASTEXITCODE"
}
