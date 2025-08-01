# WatermarkRemover-AI

**Outil de suppression de filigranes alimenté par l'IA utilisant les modèles Florence-2 et LaMA**

![Exemple de suppression de filigrane](https://raw.githubusercontent.com/D-Ogi/WatermarkRemover-AI/main/docs/images/demo.jpg)

## Aperçu

`WatermarkRemover-AI` est une application de pointe qui utilise des modèles d'IA pour détecter et supprimer les filigranes de manière précise. Elle utilise Florence-2 de Microsoft pour identifier les filigranes et LaMA pour le remplissage naturel des régions supprimées. Le logiciel propose à la fois une interface en ligne de commande (CLI) et une interface graphique (GUI) basée sur PyQt6, le rendant accessible aux utilisateurs novices et avancés.

## Caractéristiques

* **Modes multiples** : Traitez des fichiers individuels ou des dossiers entiers d'images et de vidéos.
* **Détection avancée de filigranes** : Utilise la détection à vocabulaire ouvert de Florence-2 pour une identification précise des filigranes.
* **Inpainting sans couture** : Emploie LaMA pour un remplissage de haute qualité et sensible au contexte.
* **Support vidéo** : Traitement des fichiers vidéo image par image pour supprimer les filigranes.
* **Sortie personnalisable** :  
   * Configurez la taille maximale de la boîte englobante pour la détection des filigranes.  
   * Définissez la transparence pour les régions de filigrane (images uniquement).  
   * Forcez des formats de sortie spécifiques (PNG, WEBP, JPG pour les images; MP4, AVI pour les vidéos).
* **Suivi de la progression** : Mises à jour de la progression en temps réel en mode GUI et CLI.
* **Support du mode sombre** : L'interface graphique s'adapte automatiquement aux paramètres du mode sombre du système.
* **Gestion efficace des ressources** : Optimisé pour l'accélération GPU à l'aide de CUDA (optionnel).

## Installation rapide

Consultez le fichier [DEMARRAGE_RAPIDE.md](./DEMARRAGE_RAPIDE.md) pour une installation et une mise en route rapides.

Pour une installation détaillée, référez-vous au fichier [INSTALLATION_FR.md](./INSTALLATION_FR.md).

### Prérequis

* Conda/Miniconda installé.
* CUDA (optionnel pour l'accélération GPU ; l'application fonctionne bien sur CPU également).

### Installation en un clic

Exécutez le script PowerShell d'installation :

```powershell
powershell -ExecutionPolicy Bypass -File install_windows.ps1
```

Ce script installe automatiquement toutes les dépendances et télécharge le modèle LaMA nécessaire.

## Utilisation

### Utilisation de l'interface graphique (GUI)

1. **Lancez l'application** avec :  
   ```
   conda activate py312aiwatermark
   python remwmgui.py
   ```

2. **Configurez les paramètres** :  
   * **Mode** : Sélectionnez "Process Single File" ou "Process Directory"  
   * **Chemins** : Parcourez et définissez les répertoires d'entrée/sortie  
   * **Options** : 
     * Activer l'écrasement des fichiers existants
     * Activer la transparence pour les régions de filigrane (images uniquement)
     * Ajuster la taille maximale de la boîte englobante
   * **Format de sortie** : Choisissez entre PNG, WEBP, JPG pour les images, MP4, AVI pour les vidéos, ou conserver le format d'origine

3. **Commencez le traitement** :  
   * Cliquez sur "Start" pour démarrer
   * Surveillez la progression et les logs dans l'interface

### Utilisation en ligne de commande (CLI)

1. **Commande de base** :  
   ```
   python remwm.py chemin_entrée chemin_sortie
   ```

2. **Options** :  
   * `--overwrite` : Écrase les fichiers existants
   * `--transparent` : Rend les régions de filigrane transparentes (images uniquement)
   * `--max-bbox-percent` : Définit la taille maximale de la boîte englobante (par défaut : 10%)
   * `--force-format` : Force le format de sortie (PNG, WEBP, JPG pour les images; MP4, AVI pour les vidéos)

3. **Exemples** :  
   ```
   python remwm.py ./images_entrée ./images_sortie --overwrite --max-bbox-percent=15 --force-format=PNG
   ```
   
   ```
   python remwm.py ./video_entrée.mp4 ./video_sortie.mp4 --max-bbox-percent=15 --force-format=MP4
   ```

## Remarques sur la mise à niveau

Si vous avez déjà utilisé une version antérieure du dépôt, suivez ces étapes pour mettre à niveau :

1. **Mettez à jour le dépôt** :  
   ```
   git pull
   ```

2. **Supprimez l'ancien environnement** :  
   ```
   conda deactivate
   conda env remove -n py312
   ```

3. **Exécutez le script d'installation** :  
   ```
   powershell -ExecutionPolicy Bypass -File install_windows.ps1
   ```

## Problèmes courants

Consultez le fichier [INSTALLATION_FR.md](./INSTALLATION_FR.md) pour les solutions aux problèmes courants.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails. 