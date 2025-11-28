$ErrorActionPreference = "Stop"

$CarlaExecPath = "$Env:USERPROFILE\Downloads\CARLA_0.9.16\CarlaUE4.exe"

try {
    Write-Host "Starting CarlaUE4.exe"
    Start-Process -FilePath $CarlaExecPath
    
    Write-Host "Starting train_vrrl.py"
    .venv\Scripts\python.exe train_vrrl.py
    Write-Host "Completed train_vrrl.py"
} finally {
    Write-Host "Stopping CarlaUE4.exe"
    Stop-Process -Name "CarlaUE4-Win64-Shipping"
    Write-Host "Stopped CarlaUE4.exe"
}
