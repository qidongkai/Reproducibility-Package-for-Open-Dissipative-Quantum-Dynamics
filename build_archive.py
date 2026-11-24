import os
import zipfile
from pathlib import Path
from datetime import datetime

root = Path(__file__).parent
archive_name = f"cmp_lindbladian_repro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
exclude = {archive_name, "__pycache__", ".git", ".gitignore"}

def add_folder(zf: zipfile.ZipFile, folder: Path):
    for r, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in exclude]
        for fn in files:
            fp = Path(r) / fn
            rel = fp.relative_to(root)
            zf.write(fp, arcname=rel)

with zipfile.ZipFile(archive_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for entry in root.iterdir():
        if entry.name in exclude:
            continue
        if entry.is_dir():
            add_folder(z, entry)
        else:
            z.write(entry, arcname=entry.name)
print(f"✅ Archive created: {archive_name}")