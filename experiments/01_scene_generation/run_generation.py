"""Run the prompt/repair experiment using an external worldc checkout."""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worldc-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/scenes"))
    parser.add_argument(
        "--variant", choices=["constrained", "plain"], default="constrained"
    )
    parser.add_argument("--prompt-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-repairs", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--max-budget-usd", type=float, default=20)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    worldc = args.worldc_root.resolve()
    out = args.out.resolve()
    prompt_file = worldc / "prompts/prompts.csv"
    with prompt_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.prompt_ids:
        unknown = set(args.prompt_ids) - {r["id"] for r in rows}
        if unknown:
            parser.error("Unknown prompt IDs: " + ", ".join(sorted(unknown)))
        rows = [r for r in rows if r["id"] in args.prompt_ids]
    if not rows:
        parser.error("No prompts selected")
    common = [
        "--prompts-file",
        str(out / "prompts.txt"),
        "--max-turns",
        str(args.max_turns),
        "--max-budget-usd",
        str(args.max_budget_usd),
        "--seed",
        str(args.seed),
        "--max-repairs",
        str(args.max_repairs),
        "--timeout",
        str(args.timeout),
        "--effort",
        "high",
    ]
    jobs = []
    for model in args.models:
        provider, name = (
            ("ollama", model[7:]) if model.startswith("ollama:") else ("claude", model)
        )
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        dataset = out / slug
        if provider == "ollama":
            command = ["bash", str(worldc / "scripts/generate_ollama.sh"), *common]
        else:
            command = [
                "uv",
                "run",
                "--project",
                str(worldc),
                "worldc",
                "generate",
                "--model",
                name,
                "--dataset",
                str(dataset),
                *common,
            ]
        jobs.append(
            dict(model=model, provider=provider, dataset=str(dataset), command=command)
        )
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out / "prompts.txt").write_text(
        "\n".join(r["text_" + args.variant] for r in rows) + "\n", encoding="utf-8"
    )
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                variant=args.variant,
                prompt_ids=[r["id"] for r in rows],
                seed=args.seed,
                max_repairs=args.max_repairs,
                max_turns=args.max_turns,
                max_budget_usd=args.max_budget_usd,
                timeout_s=args.timeout,
                jobs=jobs,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    results = []
    for job in jobs:
        env = os.environ.copy()
        if job["provider"] == "ollama":
            env.update(OLLAMA_MODEL=job["model"][7:], WORLDC_DATASET=job["dataset"])
        with (out / (Path(job["dataset"]).name + ".log")).open(
            "w", encoding="utf-8"
        ) as log:
            result = subprocess.run(
                job["command"],
                cwd=worldc,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        results.append(dict(model=job["model"], returncode=result.returncode))
        (out / "status.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return int(any(r["returncode"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
