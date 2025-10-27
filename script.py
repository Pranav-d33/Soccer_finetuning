import shutil
from pathlib import Path
import kagglehub

repo_dir = Path(__file__).resolve().parent
data_dir = repo_dir / "data"
data_dir.mkdir(exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("hugomathien/soccer")
print("Downloaded to:", path)

src = Path(path)
if src.exists():
    if src.is_file():
        # If it's an archive, unpack into data_dir
        if src.suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2"}:
            shutil.unpack_archive(str(src), str(data_dir))
            try:
                src.unlink()
            except Exception:
                pass
        else:
            shutil.move(str(src), str(data_dir / src.name))
    elif src.is_dir():
        # Move contents into data_dir (overwrite existing)
        for p in src.iterdir():
            dest = data_dir / p.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(p), str(dest))
else:
    print("Warning: downloaded path does not exist:", path)

print("Path to dataset files:", data_dir)