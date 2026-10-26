"""
Test bootstrap: runs the app against a temporary COPY of data/ so tests never touch real project data.
Import this module BEFORE importing `app`.
"""
import atexit
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REAL_DATA_DIR = PROJECT_DIR / "data"
TEMP_ROOT = PROJECT_DIR / "work" / "test-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

# Pristine snapshot of the shipped data, plus the working directory the app will use
_ROOT = Path(tempfile.mkdtemp(prefix="sjp-test-", dir=TEMP_ROOT))
SNAPSHOT_DIR = _ROOT / "snapshot"
DATA_DIR = _ROOT / "work"
shutil.copytree(REAL_DATA_DIR, SNAPSHOT_DIR)
shutil.copytree(SNAPSHOT_DIR, DATA_DIR)
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)


def _fingerprint(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*")) if p.is_file()}


REAL_DATA_FINGERPRINT = _fingerprint(REAL_DATA_DIR)

os.environ["SMART_JOB_PORTAL_DATA_DIR"] = str(DATA_DIR)
os.environ["APP_ENV"] = "development"
os.environ["SECRET_KEY"] = "unit-test-secret-key-not-for-production"
os.environ["ADMIN_PASSWORD"] = "test-admin-password-not-production"


def real_data_unchanged() -> bool:
    return _fingerprint(REAL_DATA_DIR) == REAL_DATA_FINGERPRINT


def fresh_work_dir() -> Path:
    """Reset the working data dir to the pristine snapshot (files only)."""
    for p in DATA_DIR.glob("*"):
        if p.is_file():
            p.unlink()
    for p in SNAPSHOT_DIR.glob("*"):
        shutil.copy2(p, DATA_DIR / p.name)
    return DATA_DIR


def copy_snapshot_to(target: Path) -> Path:
    """Make an independent data dir (used for subprocess/restart tests)."""
    target.mkdir(parents=True, exist_ok=True)
    for p in SNAPSHOT_DIR.glob("*"):
        shutil.copy2(p, target / p.name)
    return target
