# Guide d'Installation de WatermarkRemover-AI pour Windows

Ce guide vous aidera à installer et configurer WatermarkRemover-AI sur votre système Windows.

## Prérequis

1. **Python et Conda** : Vous devez avoir Miniconda ou Anaconda installé sur votre système. Si ce n'est pas le cas, téléchargez et installez Miniconda depuis [le site officiel](https://docs.conda.io/en/latest/miniconda.html).

2. **Git** : Si vous souhaitez cloner directement le dépôt, vous aurez besoin de Git. Téléchargez-le depuis [git-scm.com](https://git-scm.com/downloads).

## Méthode 1: Utilisation du script d'installation automatique

1. Ouvrez une invite de commande (CMD) dans le dossier du projet.

2. Exécutez le script d'installation:
   ```
   install_windows.bat
   ```

3. Suivez les instructions à l'écran.

## Méthode 2: Installation manuelle

Si le script automatique ne fonctionne pas, suivez ces étapes manuelles:

1. **Ouvrez une invite de commande** (CMD) ou PowerShell avec les droits administrateur.

2. **Naviguez vers le dossier du projet**:
   ```
   cd chemin\vers\WatermarkRemover-AI
   ```

3. **Créez l'environnement conda**:
   ```
   conda env create -f environment.yml
   ```

4. **Activez l'environnement**:
   ```
   conda activate py312aiwatermark
   ```

5. **Installez les dépendances supplémentaires**:
   ```
   pip install PyQt6 transformers iopaint opencv-python-headless
   ```

6. **Téléchargez le modèle LaMA**:
   ```
   iopaint download --model lama
   ```

## Lancement de l'application

Après l'installation, vous pouvez lancer l'application:

1. **Activez l'environnement** (si ce n'est pas déjà fait):
   ```
   conda activate py312aiwatermark
   ```

2. **Lancez l'application GUI**:
   ```
   python remwmgui.py
   ```

## Utilisation

Une fois l'application lancée:

1. **Choisissez le mode de traitement**:
   - Traitement d'une image unique
   - Traitement d'un dossier entier

2. **Sélectionnez les chemins d'entrée et de sortie**.

3. **Configurez les options**:
   - Activez l'écrasement des fichiers existants si nécessaire
   - Définissez si les zones de filigrane doivent être rendues transparentes
   - Ajustez la taille maximale de la boîte englobante pour la détection
   - Sélectionnez le format de sortie (PNG, WEBP, JPG ou format d'origine)

4. **Cliquez sur "Start" pour commencer le traitement**.

## Problèmes courants et solutions

### Problème: "Conda n'est pas reconnu comme une commande interne ou externe"
**Solution**: Assurez-vous que Conda est correctement installé et que son chemin est ajouté à la variable d'environnement PATH.

### Problème: Échec lors de l'installation des dépendances
**Solution**: Essayez d'exécuter les commandes d'installation individuellement et vérifiez les messages d'erreur spécifiques.

### Problème: L'application ne démarre pas
**Solution**: Vérifiez que l'environnement Python est correctement activé avec `conda activate py312aiwatermark`.

### Problème: Le modèle LaMA ne se télécharge pas
**Solution**: Assurez-vous d'avoir une connexion Internet stable et réessayez avec la commande `iopaint download --model lama`.

## Support

Si vous rencontrez des problèmes, vous pouvez:
- Ouvrir une issue sur le [dépôt GitHub](https://github.com/D-Ogi/WatermarkRemover-AI)
- Consulter les discussions existantes pour voir si quelqu'un a déjà rencontré le même problème

---

Profitez de votre nouvel outil de suppression de filigranes alimenté par l'IA! 