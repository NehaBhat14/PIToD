#!/usr/bin/env python3
"""
Run Dynamic PIToD training, regenerate main result plots, then git-commit artifacts.

Forwards all unrecognized CLI arguments to dynamic-main-TH.py. Parses -info for
plot_main_results_pitod.py --top-level-dir (defaults to SAC+ToD if omitted).

Colab: configure git before relying on commits, e.g.
  git config user.email "you@example.com"
  git config user.name "Your Name"
or set GIT_AUTHOR_EMAIL / GIT_AUTHOR_NAME.

Background example:
  nohup python -u run_dynamic_pitod_pipeline.py -env Hopper-v2 -info my_run \\
    --replay_mode dynamic_pitod -evaluate_bias 1 > pipeline.log 2>&1 &
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train with dynamic-main-TH.py, run plot_main_results_pitod.py, then git add/commit.",
    )
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
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
    args, training_argv = parser.parse_known_args()

    train_script = REPO_ROOT / "dynamic-main-TH.py"
    plot_script = REPO_ROOT / "plot_main_results_pitod.py"

    subprocess.run(
        [sys.executable, "-u", str(train_script)] + training_argv,
        cwd=str(REPO_ROOT),
        check=True,
    )

    info = _parse_info_from_training_argv(training_argv)
    top_level = info if info is not None else "SAC+ToD"

    if not args.skip_plot:
        subprocess.run(
            [sys.executable, str(plot_script), "--top-level-dir", top_level],
            cwd=str(REPO_ROOT),
            check=True,
        )

    if args.skip_git:
        return

    if not (REPO_ROOT / ".git").is_dir():
        print("No .git directory; skipping git commit.", file=sys.stderr)
        return

    if args.git_all:
        subprocess.run(["git", "add", "-A"], cwd=str(REPO_ROOT), check=True)
    else:
        for name in ("runs", "figure"):
            p = REPO_ROOT / name
            if p.exists():
                subprocess.run(["git", "add", "--", str(p)], cwd=str(REPO_ROOT), check=True)

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(REPO_ROOT),
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

    subprocess.run(["git", "commit", "-m", msg], cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
