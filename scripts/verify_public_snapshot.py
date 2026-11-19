"""Reproducible source-snapshot validation, without importing private owners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".py", ".json", ".md", ".yml", ".yaml", ".txt"}
FORBIDDEN = {".mp4", ".mov", ".png", ".jpg", ".pth", ".safetensors", ".gguf",
             ".zip", ".dll", ".exe", ".aex", ".prm", ".sqlite", ".db"}


def main() -> int:
    manifest = json.loads((ROOT / "SOURCE_SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, list) or not manifest:
        raise RuntimeError("Source mirror manifest is missing.")
    touched: set[str] = set()
    for item in manifest:
        name = item["File"]
        if not isinstance(name, str) or name in touched or ".." in Path(name).parts:
            raise RuntimeError("Manifest has duplicate or unsafe path.")
        touched.add(name)
        file = ROOT / name
        if not file.is_file() or file.is_symlink():
            raise RuntimeError(f"Missing or linked source file: {name}")
        if file.stat().st_size != item["Bytes"]:
            raise RuntimeError(f"Snapshot size changed: {name}")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest.casefold() != item["Sha256"].casefold():
            raise RuntimeError(f"Snapshot hash mismatch: {name}")
        if file.suffix == ".py":
            compile(file.read_bytes(), str(file), "exec")
        elif file.suffix == ".json":
            json.loads(file.read_text(encoding="utf-8-sig"))
    for file in ROOT.rglob("*"):
        if file.is_symlink():
            raise RuntimeError(f"Symlink is not allowed in public mirror: {file}")
        if file.is_file() and (file.suffix.casefold() in FORBIDDEN or
                              file.suffix.casefold() not in ALLOWED and
                              file.name not in {".gitignore", ".gitattributes"}):
            # No exception for multimedia fixture data or model checkpoint files.
            if ".git" not in file.parts:
                raise RuntimeError(f"Unexpected public file type: {file}")
    print(f"Validated {len(manifest)} source files, hashes and Python/JSON syntax.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
