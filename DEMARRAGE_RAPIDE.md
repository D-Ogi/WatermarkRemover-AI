# WatermarkRemover-AI - Guide de démarrage rapide

## Installation rapide sur Windows

1. **Prérequis** : Installez [Miniconda](https://docs.conda.io/en/latest/miniconda.html) si ce n'est pas déjà fait.

2. **Installez l'application** :
   - Ouvrez PowerShell en tant qu'administrateur
   - Naviguez vers le dossier du projet
   - Exécutez la commande : 
     ```
     powershell -ExecutionPolicy Bypass -File install_windows.ps1
     ```

3. **Alternative** : Installation manuelle
   ```powershell
   # Créer l'environnement conda
   conda env create -f environment.yml
   
   # Activer l'environnement
   conda activate py312aiwatermark
   
   # Installer les dépendances supplémentaires
   pip install PyQt6 transformers iopaint opencv-python-headless
   
   # Télécharger le modèle LaMA
   iopaint download --model lama
   ```

## Lancement de l'application

1. Ouvrez PowerShell
2. Activez l'environnement : `conda activate py312aiwatermark`
3. Lancez l'application : `python remwmgui.py`

## Utilisation basique

1. **Mode** : Choisissez entre le traitement d'une image unique ou d'un dossier entier.
2. **Chemins** : Définissez les chemins d'entrée et de sortie.
3. **Options** :
   - **Overwrite Existing Files** : Écraser les fichiers existants.
   - **Make Watermark Transparent** : Rendre les zones de filigrane transparentes.
   - **Max BBox Percent** : Taille maximale de la zone de filigrane (en % de l'image).
   - **Format de sortie** : PNG, WEBP, JPG ou conserver le format d'origine.
4. **Start** : Lancez le traitement.

## Conseils d'utilisation

- Pour de meilleurs résultats, utilisez le format PNG qui supporte la transparence.
- Si vous traitez de grandes images, utilisez une valeur plus faible pour Max BBox Percent.
- L'option transparence est particulièrement utile pour préserver les détails de l'image.

## En cas de problème

- Vérifiez que tous les prérequis sont installés correctement.
- Assurez-vous que l'environnement conda est activé.
- Consultez le fichier `INSTALLATION_FR.md` pour des instructions plus détaillées. 