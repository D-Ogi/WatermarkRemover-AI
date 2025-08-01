@echo off
echo ==================================
echo  Installation de WatermarkRemover-AI
echo ==================================
echo.

REM Vérification de l'installation de Conda
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez installer Miniconda ou Anaconda avant de continuer.
    echo Téléchargez-le sur : https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Conda détecté. Vérifification de l'environnement...
echo.

REM Vérification de l'existence de l'environnement
conda env list | findstr /C:"py312aiwatermark" >nul
if %ERRORLEVEL% EQU 0 (
    echo L'environnement py312aiwatermark existe déjà.
    choice /C YN /M "Voulez-vous le recréer? (Y/N)"
    if %ERRORLEVEL% EQU 1 (
        echo Suppression de l'ancien environnement...
        call conda env remove -n py312aiwatermark
    ) else (
        echo Activation de l'environnement existant...
        goto ACTIVATION
    )
)

echo Création de l'environnement conda à partir du fichier environment.yml...
call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo Erreur lors de la création de l'environnement.
    pause
    exit /b 1
)

:ACTIVATION
echo Activation de l'environnement py312aiwatermark...
call conda activate py312aiwatermark
if %ERRORLEVEL% NEQ 0 (
    echo Erreur lors de l'activation de l'environnement.
    pause
    exit /b 1
)

echo Installation des dépendances supplémentaires...
pip install PyQt6 transformers iopaint opencv-python-headless
if %ERRORLEVEL% NEQ 0 (
    echo Erreur lors de l'installation des dépendances.
    pause
    exit /b 1
)

echo Téléchargement du modèle LaMA...
iopaint download --model lama
if %ERRORLEVEL% NEQ 0 (
    echo Avertissement: Erreur lors du téléchargement du modèle LaMA.
    echo Vous pourrez réessayer plus tard avec la commande: iopaint download --model lama
)

echo.
echo ===============================
echo  Installation terminée!
echo ===============================
echo.
echo Pour lancer l'application:
echo 1. Ouvrez une invite de commande
echo 2. Activez l'environnement: conda activate py312aiwatermark
echo 3. Lancez l'application: python remwmgui.py
echo.

choice /C YN /M "Voulez-vous lancer l'application maintenant? (Y/N)"
if %ERRORLEVEL% EQU 1 (
    echo Lancement de l'application...
    python remwmgui.py
)

echo.
pause 