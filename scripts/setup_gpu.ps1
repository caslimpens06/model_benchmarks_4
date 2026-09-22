$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================="
Write-Host "Reachy Mini Benchmark - GPU setup"
Write-Host "==============================================="
Write-Host ""

Write-Host "Installing Python requirements..."
python -m pip install --upgrade pip

python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Removing any CPU-only PyTorch installation..."
python -m pip uninstall -y torch torchvision torchaudio

Write-Host ""
Write-Host "Installing PyTorch CUDA 12.6..."
python -m pip install `
    torch==2.6.0 `
    torchvision==0.21.0 `
    torchaudio==2.6.0 `
    --index-url https://download.pytorch.org/whl/cu126 `
    --force-reinstall

Write-Host ""
Write-Host "Checking CUDA..."
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

Write-Host ""

if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed."
}

Write-Host "GPU environment installed."
Write-Host ""