#!/usr/bin/env python3
"""Materialize the shared assets into each language book.

The repository keeps one copy of the language-neutral assets in ``shared/``:

    shared/helper    Python helper package imported by the notebooks
    shared/data      input CSV/JSON files used by the notebooks
    shared/_static   Jupyter Book static files (lang-switch.js, css)

The notebooks refer to these with paths relative to the book root
(``data/...``, ``import helper``), so ``polars_guide/`` and
``polars_guide_cn/`` need real directories.  Those directories are generated
by this script and are ignored by git.

``data/`` also collects files that the notebooks create while they run
(``data/stock/`` downloaded by 006_polars_input_and_output, ``*.arrow``,
generated ``*.csv``).  They are not part of ``shared/`` and are never removed
here: the sync only copies files that exist in ``shared/``.

Usage:
    python sync_assets.py                      # copy into both books
    python sync_assets.py --check              # report drift, exit 1 if any
    python sync_assets.py --book polars_guide_cn
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "shared"
BOOKS = ("polars_guide", "polars_guide_cn")
ASSET_DIRS = ("helper", "data", "_static")
# Stale files under data/ are probably notebook output, so never delete them.
PRUNE_DIRS = {"helper", "_static"}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def walk_files(base: Path) -> dict[str, Path]:
    """Return {relative posix path: absolute path}, skipping caches."""
    files: dict[str, Path] = {}
    for path in base.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(base)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files[rel.as_posix()] = path
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(src: Path, dst: Path):
    src_files = walk_files(src)
    dst_files = walk_files(dst) if dst.is_dir() else {}
    missing = sorted(set(src_files) - set(dst_files))
    extra = sorted(set(dst_files) - set(src_files))
    changed = sorted(
        name
        for name in set(src_files) & set(dst_files)
        if sha256(src_files[name]) != sha256(dst_files[name])
    )
    return src_files, missing, changed, extra


def is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def sync_book(book: str, check: bool) -> int:
    book_dir = ROOT / book
    if not book_dir.is_dir():
        print(f"skip {book}: not present")
        return 0

    drift = 0
    for name in ASSET_DIRS:
        src = SHARED / name
        dst = book_dir / name
        if not src.is_dir():
            raise SystemExit(f"missing shared directory: {src}")

        src_files, missing, changed, extra = compare(src, dst)
        prunable = name in PRUNE_DIRS

        if check:
            for item in missing:
                print(f"[missing] {book}/{name}/{item}")
            for item in changed:
                print(f"[stale]   {book}/{name}/{item}")
            if prunable:
                for item in extra:
                    print(f"[extra]   {book}/{name}/{item}")
            drift += len(missing) + len(changed)
            drift += len(extra) if prunable else 0
            continue

        if is_link(dst):
            raise SystemExit(f"refusing to write into a link: {dst}")
        dst.mkdir(parents=True, exist_ok=True)

        for item in missing:
            target = dst / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_files[item], target)
            print(f"[copy]   {book}/{name}/{item}")
        for item in changed:
            shutil.copy2(src_files[item], dst / item)
            print(f"[update] {book}/{name}/{item}")
        if prunable:
            for item in extra:
                target = dst / item
                target.unlink()
                print(f"[remove] {book}/{name}/{item}")
                parent = target.parent
                while parent != dst and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
        print(f"[ok]     {book}/{name} ({len(src_files)} files)")

    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy shared/ into each language book.",
    )
    parser.add_argument(
        "--book",
        action="append",
        choices=BOOKS,
        help="book directory to sync (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report drift, do not write",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the copies (default)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="sync every book (default)",
    )
    args = parser.parse_args(argv)

    books = args.book or list(BOOKS)
    drift = sum(sync_book(book, args.check) for book in books)
    if args.check and drift:
        print(f"\n{drift} file(s) out of sync; run: python sync_assets.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
