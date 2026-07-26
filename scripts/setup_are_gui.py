"""Install and build the native Meta ARE GUI at the protocol-pinned revision."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
ARE_REPOSITORY = "https://github.com/facebookresearch/meta-agents-research-environments.git"
ARE_REVISION = "7946367413129784139e785ae4c351090002a0bb"
SOURCE_DIR = ROOT / "artifacts" / "are-gui-source"


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _git(source_dir: Path, *arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={source_dir}", *arguments],
        cwd=source_dir,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def setup(*, source_dir: Path = SOURCE_DIR) -> Path:
    _run(["uv", "sync", "--extra", "test", "--extra", "gui"])
    if not source_dir.exists():
        _run(["git", "clone", "--filter=blob:none", ARE_REPOSITORY, str(source_dir)])
    revision = _git(source_dir, "rev-parse", "HEAD", capture_output=True).stdout.strip()
    if revision != ARE_REVISION:
        _git(source_dir, "fetch", "origin", ARE_REVISION)
        _git(source_dir, "checkout", "--detach", ARE_REVISION)
    client = source_dir / "are" / "simulation" / "gui" / "client"
    if not (client / "package.json").is_file():
        raise RuntimeError("pinned ARE checkout does not contain the GUI client")
    if shutil.which("npm") is None:
        raise RuntimeError("npm is required to build the native ARE GUI")
    _run(["npm", "ci"], cwd=client)
    _run(["npm", "run", "build"], cwd=client)
    build = client / "build"
    if not (build / "index.html").is_file():
        raise RuntimeError("ARE GUI build did not produce build/index.html")
    return build


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    args = parser.parse_args(argv)
    build = setup(source_dir=args.source_dir.resolve())
    print(f"native ARE GUI built at {build}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
