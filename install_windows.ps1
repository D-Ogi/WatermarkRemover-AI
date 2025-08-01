# Script d'installation PowerShell pour WatermarkRemover-AI
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Installation de WatermarkRemover-AI" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Vérification de Conda
try {
    $condaVersion = conda --version
    Write-Host "Conda détecté: $condaVersion" -ForegroundColor Green
}
catch {
    Write-Host "Conda n'est pas installé ou n'est pas dans le PATH." -ForegroundColor Red
    Write-Host "Veuillez installer Miniconda ou Anaconda avant de continuer." -ForegroundColor Red
    Write-Host "Téléchargez-le sur : https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    Read-Host -Prompt "Appuyez sur Entrée pour quitter"
    exit 1
}

Write-Host "Vérification de l'environnement..."

# Vérification de l'existence de l'environnement
$envExists = conda env list | Select-String "py312aiwatermark"
if ($envExists) {
    Write-Host "L'environnement py312aiwatermark existe déjà." -ForegroundColor Yellow
    $recreate = Read-Host "Voulez-vous le recréer? (o/n)"
    if ($recreate -eq "o" -or $recreate -eq "O") {
        Write-Host "Suppression de l'ancien environnement..." -ForegroundColor Yellow
        conda env remove -n py312aiwatermark
    }
    else {
        Write-Host "Activation de l'environnement existant..." -ForegroundColor Green
        $activateEnv = $true
    }
}

if (-not $activateEnv) {
    Write-Host "Création de l'environnement conda à partir du fichier environment.yml..." -ForegroundColor Cyan
    conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Erreur lors de la création de l'environnement." -ForegroundColor Red
        Read-Host -Prompt "Appuyez sur Entrée pour quitter"
        exit 1
    }
}

Write-Host "Activation de l'environnement py312aiwatermark..." -ForegroundColor Cyan
# Sous PowerShell, nous devons utiliser une approche différente
# pour activer l'environnement dans le script lui-même
$condaPath = (Get-Command conda).Source
$condaExe = Split-Path -Parent $condaPath
$activateScript = Join-Path $condaExe "..\..\shell\condabin\conda-hook.ps1"
. $activateScript
conda activate py312aiwatermark

Write-Host "Installation des dépendances supplémentaires..." -ForegroundColor Cyan
pip install PyQt6 transformers iopaint opencv-python-headless
if ($LASTEXITCODE -ne 0) {
    Write-Host "Erreur lors de l'installation des dépendances." -ForegroundColor Red
    Read-Host -Prompt "Appuyez sur Entrée pour quitter"
    exit 1
}

Write-Host "Téléchargement du modèle LaMA..." -ForegroundColor Cyan
iopaint download --model lama
if ($LASTEXITCODE -ne 0) {
    Write-Host "Avertissement: Erreur lors du téléchargement du modèle LaMA." -ForegroundColor Yellow
    Write-Host "Vous pourrez réessayer plus tard avec la commande: iopaint download --model lama" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===============================" -ForegroundColor Green
Write-Host "  Installation terminée!" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""
Write-Host "Pour lancer l'application:" -ForegroundColor Cyan
Write-Host "1. Ouvrez une invite de commande PowerShell" -ForegroundColor Cyan
Write-Host "2. Activez l'environnement: conda activate py312aiwatermark" -ForegroundColor Cyan
Write-Host "3. Lancez l'application: python remwmgui.py" -ForegroundColor Cyan
Write-Host ""

$launch = Read-Host "Voulez-vous lancer l'application maintenant? (o/n)"
if ($launch -eq "o" -or $launch -eq "O") {
    Write-Host "Lancement de l'application..." -ForegroundColor Green
    python remwmgui.py
}

Write-Host ""
Read-Host -Prompt "Appuyez sur Entrée pour quitter" 