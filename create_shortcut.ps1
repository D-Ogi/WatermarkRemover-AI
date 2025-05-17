$WshShell = New-Object -ComObject WScript.Shell

# Chemin complet vers le dossier du projet
$projectPath = (Get-Item -Path ".").FullName

# Créer un raccourci pour le script batch
$shortcutBat = $WshShell.CreateShortcut("$projectPath\WatermarkRemover-AI Environnement (CMD).lnk")
$shortcutBat.TargetPath = "cmd.exe"
$shortcutBat.Arguments = "/k `"$projectPath\activate_env.bat`""
$shortcutBat.WorkingDirectory = $projectPath
$shortcutBat.IconLocation = "C:\Windows\System32\cmd.exe,0"
$shortcutBat.Description = "Ouvre l'environnement conda pour WatermarkRemover-AI (CMD)"
$shortcutBat.Save()

# Créer un raccourci pour le script PowerShell
$shortcutPs = $WshShell.CreateShortcut("$projectPath\WatermarkRemover-AI Environnement (PowerShell).lnk")
$shortcutPs.TargetPath = "powershell.exe"
$shortcutPs.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$projectPath\activate_env.ps1`""
$shortcutPs.WorkingDirectory = $projectPath
$shortcutPs.IconLocation = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe,0"
$shortcutPs.Description = "Ouvre l'environnement conda pour WatermarkRemover-AI (PowerShell)"
$shortcutPs.Save()

# Créer un raccourci direct pour l'application GUI
$shortcutGui = $WshShell.CreateShortcut("$projectPath\WatermarkRemover-AI (GUI).lnk")
$shortcutGui.TargetPath = "cmd.exe"
$shortcutGui.Arguments = "/k `"conda activate py312aiwatermark && python $projectPath\remwmgui.py`""
$shortcutGui.WorkingDirectory = $projectPath
$shortcutGui.Description = "Lance directement l'interface graphique de WatermarkRemover-AI"
$shortcutGui.Save()

Write-Host "Raccourcis créés avec succès dans le dossier $projectPath" -ForegroundColor Green
Write-Host "Vous pouvez les déplacer sur votre bureau ou dans le menu Démarrer." -ForegroundColor Yellow 