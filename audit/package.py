"""Build the public study ZIP from committed files and verified model weights.

Run `python audit/package.py --output-dir /path/to/dist` at the release commit.
The same command in an extracted ZIP rebuilds it from RELEASE-MANIFEST.json.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "MORPH-replication.zip"
MANIFEST = "RELEASE-MANIFEST.json"
VERIFICATION = "results/verification.json"
EXTERNAL_WEIGHTS = {
    "results/lm/prototype/adaptive-11-512.safetensors",
    "results/lm/prototype/adaptive-11-2048.safetensors",
}
REQUIRED = {
    "README.md", "MODEL_CARD.md", "LICENSE", "LICENSING.md", "CONTRIBUTING.md",
    "CITATION.cff", "THIRD_PARTY.md", "LICENSES/CC-BY-4.0.txt",
    "LICENSES/DejaVu.txt", "LICENSES/Karpathy-MIT.txt",
    "requirements.txt", "reproduce.py", "audit/package.py",
    "paper/MORPH.pdf", "paper/MORPH.tex", "results/fresh_replication.json",
    "results/reasoning/opt-summary.json", "results/reasoning/long-summary.json",
    "results/joint/fits.csv", "results/mechanisms/grouping.csv", VERIFICATION,
}
WEIGHT_SUFFIXES = {".safetensors", ".npz"}
STORED_SUFFIXES = WEIGHT_SUFFIXES | {".gz", ".pdf", ".png"}
PUBLIC_LOGS = {
    "results/joint/training.log", "results/lm/confirmation.log",
    "results/lm/prototype.log", "results/lm_followup/capstone.log",
    "results/mechanisms/extension.log", "results/mechanisms/grouping.log",
    "results/mechanisms/memory.log", "results/mechanisms/routing.log", "results/lm/verify.log",
    "results/prototype_controls/run.log", "results/reasoning/extended.log",
    "results/reasoning/optimized.log", "results/reasoning/sensitivity.log",
    "results/reasoning/suite.log", "results/reasoning/usability.log",
}
PRIVATE_PARTS = {
    ".git", ".venv", ".ruff_cache", ".pytest_cache", ".mypy_cache",
    "__pycache__", ".DS_Store", ".env", "dist",
}


def public_path(name):
    path = PurePosixPath(name)
    if (not name or name != path.as_posix() or path.is_absolute()
            or ".." in path.parts or "\\" in name or any(ord(c) < 32 for c in name)):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return not (
        set(path.parts) & PRIVATE_PARTS or path.parts[:2] == ("paper", "build")
        or name in {"docs/ledger.md", MANIFEST} or path.suffix == ".pyc"
        or (path.suffix == ".log" and name not in PUBLIC_LOGS)
        or any(part.startswith(".env.") and part != ".env.example" for part in path.parts)
    )


def local_file(name):
    path = ROOT / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Expected a regular release file: {name}")
    return partial(path.open, "rb")


def git_snapshot(stack, scratch):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=ROOT, text=True,
    ).strip()
    tar_path = scratch / "snapshot.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar_path), commit],
                   cwd=ROOT, check=True)
    archive = stack.enter_context(tarfile.open(tar_path, "r:"))
    files = {}
    for entry in archive:
        if entry.isdir() or not public_path(entry.name):
            continue
        if not entry.isfile():
            raise ValueError(f"Unsupported tracked file type: {entry.name}")
        mode = 0o755 if entry.mode & 0o111 else 0o644
        files[entry.name] = (mode, partial(archive.extractfile, entry))
    # The archive must carry the recipe actually used to build it.
    with files["audit/package.py"][1]() as archived_script:
        if archived_script.read() != Path(__file__).read_bytes():
            raise ValueError("The running audit/package.py differs from HEAD. Build using the committed release recipe.")
    for name in EXTERNAL_WEIGHTS - files.keys():
        files[name] = (0o644, local_file(name))
    return commit, files, None


def unpacked_snapshot():
    path = ROOT / MANIFEST
    if not path.is_file():
        raise ValueError(f"Build in the release Git checkout or an extracted archive with {MANIFEST}.")
    manifest = json.loads(path.read_bytes())
    if manifest.get("format") != 1 or not re.fullmatch(r"[0-9a-f]{40,64}", manifest["source_commit"]):
        raise ValueError("Invalid release manifest header")
    records = manifest["files"]
    files = {}
    for name, record in records.items():
        if not public_path(name) or record["mode"] not in (0o644, 0o755):
            raise ValueError(f"Invalid release manifest entry: {name}")
        files[name] = (record["mode"], local_file(name))
    return manifest["source_commit"], files, records


def entry_info(name, mode):
    entry = zipfile.ZipInfo("MORPH/" + name, (1980, 1, 1, 0, 0, 0))
    entry.create_system = 3
    entry.external_attr = (0o100000 | mode) << 16
    if PurePosixPath(name).suffix not in STORED_SUFFIXES:
        entry.compress_type = zipfile.ZIP_DEFLATED
    return entry


def copy_and_hash(source, dest, name):
    digest = hashlib.sha256()
    size = 0
    tail = b""
    while chunk := source.read(1024 * 1024):
        if name in PUBLIC_LOGS:
            if b"/Users/" in tail + chunk:
                raise ValueError(f"Unredacted local path in release log: {name}")
            tail = chunk[-5:]
        dest.write(chunk)
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temp, ExitStack() as stack:
        scratch = Path(temp)
        if (ROOT / ".git").exists():
            commit, files, expected = git_snapshot(stack, scratch)
        else:
            commit, files, expected = unpacked_snapshot()
        missing = (REQUIRED | PUBLIC_LOGS) - files.keys()
        if missing:
            raise ValueError(f"Incomplete release snapshot: {sorted(missing)}")
        with files[VERIFICATION][1]() as source:
            weights = json.load(source)["weights"]
        present = {name for name in files if PurePosixPath(name).suffix in WEIGHT_SUFFIXES}
        if len(weights) != 12 or set(weights) != present or not EXTERNAL_WEIGHTS <= present:
            raise ValueError("Release must contain the twelve checkpoints inventoried in results/verification.json")

        target = scratch / ARCHIVE
        records = {}
        with zipfile.ZipFile(target, "w") as archive:
            for name, (mode, opener) in sorted(files.items()):
                with opener() as source, archive.open(entry_info(name, mode), "w") as dest:
                    size, digest = copy_and_hash(source, dest, name)
                record = {"bytes": size, "sha256": digest}
                if name in weights and record != weights[name]:
                    raise ValueError(f"Checkpoint differs from the verified study: {name}")
                record["mode"] = mode
                if expected is not None and record != expected[name]:
                    raise ValueError(f"File differs from the release manifest: {name}")
                records[name] = record
            metadata = {"format": 1, "source_commit": commit, "files": records}
            archive.writestr(entry_info(MANIFEST, 0o644),
                             json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(target) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Archive CRC validation failed: {bad}")
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        checksum = scratch / "SHA256SUMS"
        checksum.write_text(f"{digest}  {ARCHIVE}\n", encoding="ascii")
        size = target.stat().st_size
        target.replace(output / ARCHIVE)
        checksum.replace(output / "SHA256SUMS")
    print(f"{len(records) + 1} files, 12 verified checkpoints, {size:,} bytes, SHA-256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    build(parser.parse_args().output_dir.expanduser().resolve())
