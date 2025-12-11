Write-Host "Starting CALI NeuroPatch Prototype..." -ForegroundColor Cyan

# 1. Start Cloud Server (Backend)
Write-Host "1. Launching Cloud Server (Port 5000)..." -ForegroundColor Green
Start-Process python -ArgumentList "cloud_server.py" -WorkingDirectory "cloud_infrastructure" -WindowStyle Normal

# Wait for server to initialize
Start-Sleep -Seconds 5

# 2. Start Base Station (Receiver)
Write-Host "2. Launching Receiver Hub (Port 5555)..." -ForegroundColor Green
# base_station.py imports? It uses standard libs.
# Let's ensure it runs in its dir so any local file references work
Start-Process python -ArgumentList "base_station.py" -WorkingDirectory "receiver_hub" -WindowStyle Normal

Start-Sleep -Seconds 2

# 3. Start Device Simulation (Sensor)
Write-Host "3. Launching Device Simulation (Sends to Port 5555)..." -ForegroundColor Green
# neuro_patch_sim.py relies on "../ml_pipeline" and "../WESAD", so it MUST run from device_simulation dir
Start-Process python -ArgumentList "neuro_patch_sim.py" -WorkingDirectory "device_simulation" -WindowStyle Normal

# 4. Open Dashboard
Write-Host "4. Opening Dashboard..." -ForegroundColor Yellow
Start-Process "dashboard/index.html"

Write-Host "Prototype Running!" -ForegroundColor Cyan
Write-Host "Check the opened terminal windows for logs."
Write-Host "Cloud Server: Shows Stress Predictions"
Write-Host "Receiver Hub: Shows Packet forwarding"
Write-Host "Device Sim: Shows Data transmission"
