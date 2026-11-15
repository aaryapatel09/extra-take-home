import argparse
import json
import sys
from pathlib import Path

from .schema import Applicant, InputError, Role, Settings


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise InputError(f"invalid JSON on line {line}: {exc.msg}") from exc
    return records


def save(path, content):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic writes prevent leaving an incomplete report if the process is interrupted.
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as f:
        temp = Path(f.name)
        try:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local applicant evidence review")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-role", help="Draft profiles using local Ollama")
    generate.add_argument("--description", required=True, help="UTF-8 job-description file")
    generate.add_argument("--model", default="llama3.2")
    generate.add_argument("--ollama-url", default="http://localhost:11434")
    generate.add_argument("--output", required=True)
    for name in ("evaluate", "benchmark"):
        p = commands.add_parser(name)
        p.add_argument("--role", required=True)
        p.add_argument("--applicants", required=True, help="JSONL evidence records")
        p.add_argument("--output", required=True, help="JSON report path")
        p.add_argument("--html", help="Optional standalone evidence report")
        p.add_argument("--budget", type=int, default=12)
        p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
        p.add_argument("--offline", action="store_true", help="Use only downloaded model weights")
        p.add_argument("--evidence-threshold", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=3)
        if name == "benchmark":
            p.add_argument("--split", choices=("dev", "test"), default="test")
            p.add_argument("--paraphrases", help="Optional paired semantic rewrite JSONL")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-role":
            from .generation import generate_role

            data = generate_role(
                Path(args.description).read_text(encoding="utf-8"), args.model, args.ollama_url
            )
        else:
            from .engine import evaluate, review_queue
            from .models import LocalModels

            role = Role.from_dict(read_json(args.role))
            records = read_jsonl(args.applicants)
            # Validate all input before downloading or loading models.
            applicants = [Applicant.from_dict(r) for r in records]
            if args.budget < 0:
                raise InputError("budget must be nonnegative")
            settings = Settings(
                evidence_threshold=args.evidence_threshold, evidence_top_k=args.top_k
            )
            models = LocalModels(device=args.device, offline=args.offline)
            if args.command == "benchmark":
                from .benchmark import run_benchmark

                data = run_benchmark(
                    role,
                    records,
                    models,
                    settings,
                    args.budget,
                    args.split,
                    read_jsonl(args.paraphrases) if args.paraphrases else None,
                )
                report = data["evaluation"]
            else:
                data = evaluate(role, applicants, models, settings)
                data["review_queue"] = review_queue(data["results"], args.budget)
                report = data
            if args.html:
                from .report import render_html

                save(args.html, render_html(report))
        save(args.output, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
