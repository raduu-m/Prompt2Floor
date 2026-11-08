"""Launch navigation experiment entry points from the external worldc checkout."""

import argparse
import subprocess
from pathlib import Path

ENTRYPOINTS = {
    "select": "e3_select_scenes.py",
    "run": "e3_suite.py",
    "analyze": "e3_analysis.py",
    "isaac": "e3_isaac_suite.py",
    "analyze-isaac": "e3_isaac_analysis.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worldc-root", type=Path, required=True)
    parser.add_argument("command", choices=ENTRYPOINTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.worldc_root.resolve()
    entrypoint = root / "scripts" / ENTRYPOINTS[args.command]
    if not entrypoint.is_file():
        parser.error(
            "Navigation entry point unavailable; apply integration/worldc.patch to the dependency checkout."
        )
    return subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(root),
            "python",
            str(entrypoint),
            *args.arguments,
        ]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
