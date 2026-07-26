"""Verify an already-created upstream checkout without changing it."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence


LOCKED_ARE_COMMIT = "7946367413129784139e785ae4c351090002a0bb"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
GitRunner = Callable[..., subprocess.CompletedProcess[str]]


class CheckoutVerificationError(ValueError):
    """Raised when a checkout is missing, mismatched, or dirty."""


@dataclass(frozen=True)
class CheckoutVerification:
    checkout_path: str
    commit: str
    clean: bool


def _git(
    checkout_path: Path,
    args: Sequence[str],
    *,
    git_runner: GitRunner,
) -> str:
    result = git_runner(
        ["git", *args],
        cwd=str(checkout_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git command failed").strip()
        raise CheckoutVerificationError(detail)
    return result.stdout.strip()


def verify_checkout(
    checkout_path: str | Path,
    expected_commit: str = LOCKED_ARE_COMMIT,
    *,
    git_runner: GitRunner = subprocess.run,
) -> CheckoutVerification:
    """Check HEAD and porcelain status; never fetches, checks out, or resets."""

    path = Path(checkout_path).expanduser()
    if not path.is_dir():
        raise CheckoutVerificationError(f"checkout path is not a directory: {path}")
    if not _COMMIT_RE.fullmatch(expected_commit):
        raise CheckoutVerificationError("expected_commit must be a 40-character lowercase SHA")

    commit = _git(path, ["rev-parse", "--verify", "HEAD"], git_runner=git_runner)
    if commit != expected_commit:
        raise CheckoutVerificationError(
            f"checkout commit mismatch: expected {expected_commit}, found {commit}"
        )
    status = _git(
        path,
        ["status", "--porcelain", "--untracked-files=all"],
        git_runner=git_runner,
    )
    if status:
        raise CheckoutVerificationError("checkout is not clean")
    return CheckoutVerification(str(path.resolve()), commit, True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout_path", type=Path)
    parser.add_argument("--commit", default=LOCKED_ARE_COMMIT)
    args = parser.parse_args(argv)
    try:
        result = verify_checkout(args.checkout_path, args.commit)
    except CheckoutVerificationError as error:
        parser.error(str(error))
    print(f"verified {result.checkout_path} at {result.commit}; clean={result.clean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
