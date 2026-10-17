"""Build a personal GitHub upload archive; never commit, push or open a PR."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def candidates():
    listing = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
    ], cwd=ROOT).decode().split("\0")
    files = []
    for name in sorted(set(listing) - {""}):
        path = Path(name)
        if name in {"UPLOAD_MANIFEST.json", ".github/workflows/pr_validation.yml"}:
            # The upstream competition PR workflow is not part of a personal
            # research upload. A previously generated manifest is rebuilt below.
            continue
        blocked = {".git", ".venv", "runs", "__pycache__", ".pytest_cache", ".ruff_cache", "exports", ".upload-check"}
        if blocked.intersection(path.parts):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        if path.suffix in {".log", ".pyc", ".zip"}:
            continue
        absolute = ROOT / path
        if absolute.is_symlink():
            raise RuntimeError(f"Refusing symlink in upload archive: {name}")
        if absolute.is_file():
            files.append(path)
    return files


def known_secrets():
    # Never print values. Detect accidental copies of local keys in selected files.
    path = ROOT / ".env"
    values = []
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 8:
                values.append(value.encode())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("Output already exists; choose a new name to preserve the previous archive.")
    files = candidates()
    if output in [ROOT / path for path in files]:
        raise SystemExit("Archive output overlaps an input file.")
    secrets = known_secrets()
    for path in files:
        content = (ROOT / path).read_bytes()
        if any(secret in content for secret in secrets):
            raise SystemExit(f"Local credential detected in selected file: {path}")
        if len(content) >= 100 * 1024 * 1024:
            raise SystemExit(f"File exceeds GitHub's regular file limit: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "personal_research_upload", "files": [],
                "excludes": ["Git metadata", "local runs", "credentials", "virtual environment", "caches"],
                "git_mutations": False, "network_calls": False}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            content = (ROOT / path).read_bytes()
            archive.writestr(path.as_posix(), content)
            manifest["files"].append({"path": path.as_posix(), "bytes": len(content),
                                      "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("UPLOAD_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Created {output}: {len(files)} files, {output.stat().st_size} bytes; local credential scan passed.")


if __name__ == "__main__":
    main()
