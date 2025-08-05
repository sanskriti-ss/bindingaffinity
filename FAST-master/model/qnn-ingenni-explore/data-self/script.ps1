Get-ChildItem -Path . -File | Where-Object { $_.Name.StartsWith("best_checkpoint_") } | ForEach-Object {
    $newName = $_.Name.Substring("best_checkpoint_".Length)
    Rename-Item -Path $_.FullName -NewName $newName
}