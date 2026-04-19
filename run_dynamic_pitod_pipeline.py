#!/usr/bin/env python3
"""
Run Dynamic PIToD training, regenerate main result plots, then git add / commit / push.

Forwards all unrecognized CLI arguments to dynamic-main-TH.py. Parses -info for
plot_main_results_pitod.py --top-level-dir (defaults to SAC+ToD if omitted).

After a successful commit, pushes to origin: branch ``main`` if HEAD is ``main``,
otherwise pushes the current branch to the same name on origin.

Colab: configure git and auth (HTTPS token or SSH) before push, e.g.
  git config user.email "you@example.com"
  git config user.name "Your Name"
"""

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent


def _parse_info_from_training_argv(argv: List[str]) -> Optional[str]:
    for i, tok in enumerate(argv):
        if tok == "-info" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _git_current_branch(cwd: str) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return out.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train with dynamic-main-TH.py, run plot_main_results_pitod.py, "
        "then git add / commit / push.",
    )
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Commit locally but do not git push.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Git commit message (default: timestamp + short hash of training args).",
    )
    parser.add_argument(
        "--git-all",
        action="store_true",
        help="Stage entire repo (git add -A) instead of only runs/ and figure/.",
    )
    parser.add_argument(
        "--push-remote",
        default="origin",
        help="Remote name for git push (default: origin).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print train/plot/git/push steps and exit 0 without executing them.",
    )
    args, training_argv = parser.parse_known_args()

    train_script = REPO_ROOT / "dynamic-main-TH.py"
    plot_script = REPO_ROOT / "plot_main_results_pitod.py"

    info = _parse_info_from_training_argv(training_argv)
    top_level = info if info is not None else "SAC+ToD"

    train_cmd = [sys.executable, "-u", str(train_script)] + training_argv
    plot_cmd = [sys.executable, str(plot_script), "--top-level-dir", top_level]

    if args.dry_run:
        print("[dry-run] cwd:", REPO_ROOT)
        print("[dry-run] train:", " ".join(train_cmd))
        if args.skip_plot:
            print("[dry-run] plot: (skipped --skip-plot)")
        else:
            print("[dry-run] plot:", " ".join(plot_cmd))
        if args.skip_git:
            print("[dry-run] git: (skipped --skip-git)")
        elif not (REPO_ROOT / ".git").is_dir():
            print("[dry-run] git: (would skip: no .git)")
        else:
            if args.git_all:
                print("[dry-run] git: git add -A && git commit -m <msg>")
            else:
                print("[dry-run] git: git add runs figure (if present) && git commit -m <msg>")
            if args.skip_push:
                print("[dry-run] push: (skipped --skip-push)")
            else:
                print(
                    "[dry-run] push: git push",
                    args.push_remote,
                    "main (if HEAD is main) else git push -u",
                    args.push_remote,
                    "<current-branch>",
                )
        print("[dry-run] resolved --top-level-dir:", top_level)
        return

    subprocess.run(
        train_cmd,
        cwd=str(REPO_ROOT),
        check=True,
    )

    if not args.skip_plot:
        subprocess.run(
            plot_cmd,
            cwd=str(REPO_ROOT),
            check=True,
        )

    if args.skip_git:
        return

    if not (REPO_ROOT / ".git").is_dir():
        print("No .git directory; skipping git commit.", file=sys.stderr)
        return

    cwd = str(REPO_ROOT)

    if args.git_all:
        subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    else:
        for name in ("runs", "figure"):
            p = REPO_ROOT / name
            if p.exists():
                subprocess.run(["git", "add", "--", str(p)], cwd=cwd, check=True)

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=cwd,
    )
    if staged.returncode == 0:
        print("Nothing to commit (empty index after git add).")
        return

    msg = args.commit_message
    if not msg:
        arg_blob = " ".join(training_argv).encode("utf-8", errors="replace")
        digest = hashlib.sha256(arg_blob).hexdigest()[:12]
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = f"pitod: train+plot {ts} ({digest})"

    subprocess.run(["git", "commit", "-m", msg], cwd=cwd, check=True)

    if args.skip_push:
        return

    branch = _git_current_branch(cwd)
    if branch == "main":
        subprocess.run(
            ["git", "push", "-u", args.push_remote, "main"],
            cwd=cwd,
            check=True,
        )
    else:
        print(
            f"Not on branch main (HEAD is {branch!r}); pushing this branch to "
            f"{args.push_remote} instead of main.",
            file=sys.stderr,
        )
        subprocess.run(
            ["git", "push", "-u", args.push_remote, branch],
            cwd=cwd,
            check=True,
        )


if __name__ == "__main__":
    main()
