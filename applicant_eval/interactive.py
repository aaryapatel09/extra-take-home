"""Interactive applicant reviewer: type a job, inspect rows, and save human notes."""

import argparse
import json
import logging
import os
import shlex
import unicodedata
from contextlib import nullcontext
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .engine import evaluate, review_queue
from .generation import generate_role
from .io import read_json, save
from .models import LocalModels
from .report import render_html
from .schema import InputError, Role
from .session import SessionStore, digest, load_applicants

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/applicants/sample_applicants.jsonl"
DEMO_ROLE = ROOT / "examples/applicants/backend_role.json"


def safe_text(value):
    """Display text literally; strip terminal escape and bidi control characters."""
    return "".join(
        c for c in str(value) if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cf", "Cs"}
    )


def one_path(raw):
    parts = shlex.split(raw)
    if len(parts) != 1:
        raise InputError("Supply one path; put paths containing spaces in quotes.")
    return Path(parts[0]).expanduser()


class Reviewer:
    def __init__(
        self,
        console,
        store,
        *,
        model="llama3.2:3b",
        url="http://127.0.0.1:11434",
        offline=False,
        budget=12,
        model_factory=LocalModels,
        generator=generate_role,
    ):
        if budget < 0:
            raise InputError("Review budget must be nonnegative.")
        self.console, self.store = console, store
        self.model_name, self.url, self.offline, self.budget = model, url, offline, budget
        self.model_factory, self.generator = model_factory, generator
        self.models = None
        self.applicants = []
        self.input_path = None
        self.role = None
        self.role_data = None
        self.description = None
        self.report = None

    def say(self, message="", style=None):
        self.console.print(Text(safe_text(message), style=style))

    def busy(self, message):
        if self.console.is_terminal:
            return self.console.status(message, spinner="dots")
        self.say(message, "dim")
        return nullcontext()

    def load(self, path):
        applicants = load_applicants(path)
        self.applicants, self.input_path, self.report = applicants, Path(path), None
        self.say(f"Loaded {len(applicants)} applicants from {Path(path).name}.", "cyan")

    def use_role(self, data, description=None):
        role = Role.from_dict(data)
        self.role, self.role_data, self.description, self.report = role, data, description, None

    def job(self, description, fresh=False):
        if not description.strip():
            raise InputError("Type a job description, or use 'paste' for multiple lines.")
        data = None
        if not fresh:
            try:
                data = self.store.cached_role(description, self.model_name)
            except (ValueError, OSError) as exc:
                self.say(
                    f"Couldn't use the profile cache: {exc}. Generating a new draft.", "yellow"
                )
        if data is None:
            with self.busy("Drafting qualifying profiles with your local model…"):
                data = self.generator(description, self.model_name, self.url)
            self.store.save_role(description, self.model_name, data)
        else:
            self.say("Using cached profiles. 'fresh' regenerates them.", "dim")
        self.use_role(data, description)
        self.profiles(detail=False)
        self.run()

    def run(self):
        if not self.role:
            raise InputError("Type a job description, 'role PATH', or 'demo' first.")
        if not self.applicants:
            raise InputError("Load applicants with 'load PATH' first.")
        with self.busy("Reading applicant evidence locally…"):
            if self.models is None:
                self.models = self.model_factory(offline=self.offline)
            result = evaluate(self.role, self.applicants, self.models)
        self.report = result
        self.refresh_reviews()
        self.show()

    def refresh_reviews(self):
        if not self.report:
            return
        reviews = self.store.reviews(self.role, self.applicants)
        self.report["review_queue"] = review_queue(self.report["results"], self.budget)
        for result in self.report["results"]:
            result.pop("human_review", None)
            if result["applicant_id"] in reviews:
                result["human_review"] = reviews[result["applicant_id"]]

    def require_results(self):
        if not self.report:
            raise InputError("No results yet. Type a job description or 'run'.")

    def show(self):
        self.require_results()
        self.say(f"\n{self.role.title}", "bold cyan")
        table = Table(box=None, padding=(0, 1), expand=False)
        for title in ("#", "Applicant", "Evidence", "Closest path", "Your review"):
            table.add_column(title, overflow="fold")
        for i, result in enumerate(self.report["results"], 1):
            supported = result["review_status"] == "evidence_supported"
            label = "supported" if supported else "review needed"
            table.add_row(
                str(i),
                Text(safe_text(result["applicant_id"])),
                Text(label, style="green" if supported else "yellow"),
                Text(
                    safe_text(result["closest_profile"])
                    if any(c["sources"] for c in result["evidence"])
                    else "—"
                ),
                Text(safe_text(result.get("human_review", {}).get("status", "—"))),
            )
        self.console.print(table)
        queue = self.report["review_queue"]
        reserve = sum(q["reason"] == "exploration_review" for q in queue)
        self.say(
            f"\nReview queue: {len(queue)} applicants, including {reserve} exploration {'pick' if reserve == 1 else 'picks'}.",
            "dim",
        )
        self.say(
            "Evidence support is a model assessment, not a hiring decision. Your marks are saved separately.",
            "dim",
        )
        self.say(
            "Enter a row number for evidence · mark 2 shortlist [note] · profiles · export · help",
            "cyan",
        )

    def result(self, row):
        self.require_results()
        try:
            number = int(row)
        except ValueError as exc:
            raise InputError("Use the row number shown in the table.") from exc
        if not 1 <= number <= len(self.report["results"]):
            raise InputError(f"Pick a row from 1 to {len(self.report['results'])}.")
        return self.report["results"][number - 1]

    def inspect(self, row):
        result = self.result(row)
        self.say(f"\n{result['applicant_id']}", "bold cyan")
        self.say(
            f"Closest path: {result['closest_profile']} · best-supported path: {result['best_supported_profile']}"
        )
        self.say(
            "Quotes below are applicant claims; their authenticity has not been verified.", "dim"
        )
        self.say("Retrieved quotes include context; their model support scores can differ.", "dim")
        for criterion in result["evidence"]:
            self.say(f"\n{criterion['description']} — {criterion['status']}", "bold")
            if not criterion["sources"]:
                self.say("  No evidence supplied.", "dim")
            for source in criterion["sources"]:
                self.say(f"  “{source['quote']}”")
                self.say(
                    f"  Source {source['evidence_id']}, characters {source['start']}–{source['end']}",
                    "dim",
                )
                self.say(
                    f"  Model support {source['support_score']:.3f} · contradiction {source['contradiction_score']:.3f}",
                    "dim",
                )
                if source["context_flags"]:
                    self.say("  Check context: " + ", ".join(source["context_flags"]), "yellow")
        if result.get("human_review"):
            review = result["human_review"]
            self.say(f"\nYour review: {review['status']} — {review['note']}", "cyan")

    def profiles(self, detail=True):
        if not self.role:
            raise InputError("No job profiles yet. Type a job description or 'demo'.")
        self.say(f"\n{self.role.title} · {len(self.role.profiles)} qualifying paths", "bold cyan")
        self.say(
            "Common requirements: "
            + "; ".join(c.description for c in self.role.criteria if c.required)
        )
        for criterion in self.role_data["criteria"]:
            if criterion.get("source_quote"):
                self.say(
                    f"  Job wording for {criterion['id']}: “{criterion['source_quote']}”", "dim"
                )
        for p in self.role.profiles:
            self.say(f"\n{p.id}: {p.description}", "bold")
            self.say(f"  {len(p.examples)} example profiles")
            if detail:
                for example in p.examples:
                    self.say(f"  • {example}")
                for example in p.near_misses:
                    self.say(f"  Near miss: {example}")
        if not detail:
            self.say("Type 'profiles' to inspect every example.", "dim")
        if self.role_data.get("provenance", {}).get("type") == "synthetic_draft":
            self.say(
                "\nGenerated draft: check that these requirements match the actual job. 'role PATH' loads an edited role.",
                "yellow",
            )

    def mark(self, row, status, note=""):
        result = self.result(row)
        applicant = next(a for a in self.applicants if a.id == result["applicant_id"])
        self.store.mark(self.role, applicant, status, note)
        self.refresh_reviews()
        self.say(f"Saved your review: {applicant.id} → {status}.", "green")

    def clear(self, row):
        result = self.result(row)
        applicant = next(a for a in self.applicants if a.id == result["applicant_id"])
        self.store.clear(self.role, applicant)
        self.refresh_reviews()
        self.say(f"Cleared your review for {applicant.id}.", "cyan")

    def export(self, raw):
        self.require_results()
        directory = (
            one_path(raw)
            if raw.strip()
            else self.store.directory / "exports" / digest(self.role_data)[:12]
        )
        self.refresh_reviews()
        save(
            directory / "applicants.json",
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        save(directory / "applicants.html", render_html(self.report))
        save(
            directory / "role.json", json.dumps(self.role_data, indent=2, ensure_ascii=False) + "\n"
        )
        self.say(f"Saved reports and role to {directory.resolve()}", "green")

    def help(self):
        self.say("""\nType a job description to generate profiles and evaluate the loaded applicants.
  demo                       Try the included applicants and role; no Ollama needed
  paste                      Enter a multiline job description; finish with a single .
  job DESCRIPTION            Start a different job
  load PATH                  Load JSONL/JSON, a text resume, or a folder of text resumes
  role PATH                  Use an existing role JSON file
  profiles                   Inspect the current qualifying paths and examples
  run / fresh                Evaluate again / regenerate profiles for the last job
  list / NUMBER              Show the table / inspect one applicant's evidence
  mark NUMBER STATUS [NOTE]  Save your review: shortlist, hold, or pass
  clear NUMBER               Remove your review mark
  queue / budget NUMBER      Show the review queue / change its size
  export [DIRECTORY]         Save JSON + HTML reports, including your review notes
  help / quit                Show commands / exit
Quote paths containing spaces. Reviews are scoped to the exact job and applicant text.""")

    def command(self, raw, input_fn=input):
        raw = raw.strip()
        if not raw:
            return True
        word, _, rest = raw.partition(" ")
        command = word.lower()
        if command in {"quit", "q", "exit"}:
            return False
        if command in {"help", "?"}:
            self.help()
        elif command == "demo":
            self.load(SAMPLE)
            self.use_role(read_json(DEMO_ROLE))
            self.run()
        elif command == "paste":
            self.say("Paste the job description; enter a single . on a new line to finish.")
            lines = []
            while (line := input_fn("  " if self.console.is_terminal else "")) != ".":
                lines.append(line)
            self.job("\n".join(lines))
        elif command == "job":
            self.job(rest)
        elif command == "load":
            self.load(one_path(rest))
            if self.role:
                self.run()
        elif command == "role":
            self.use_role(read_json(one_path(rest)))
            self.profiles()
            self.run()
        elif command == "profiles":
            self.profiles()
        elif command == "run":
            self.run()
        elif command == "fresh":
            if self.description is None:
                raise InputError(
                    "This role was loaded from a file. Type a job description before using 'fresh'."
                )
            self.job(self.description, fresh=True)
        elif command in {"list", "ls"}:
            self.show()
        elif command in {"inspect", "show"}:
            self.inspect(rest)
        elif command == "mark":
            parts = rest.split(maxsplit=2)
            if len(parts) < 2:
                raise InputError("Use: mark NUMBER shortlist|hold|pass [note]")
            self.mark(parts[0], parts[1].lower(), parts[2] if len(parts) > 2 else "")
        elif command == "clear":
            self.clear(rest)
        elif command == "budget":
            value = int(rest)
            if value < 0:
                raise InputError("Budget must be zero or greater.")
            self.budget = value
            self.refresh_reviews()
            self.say(f"Review budget set to {value}.")
        elif command == "queue":
            self.require_results()
            for item in self.report["review_queue"]:
                self.say(f"{item['applicant_id']} — {item['reason'].replace('_', ' ')}")
        elif command == "export":
            self.export(rest)
        elif raw.isdecimal():
            self.inspect(raw)
        else:
            self.job(raw)
        return True


def main(
    argv=None, *, input_fn=input, console=None, model_factory=LocalModels, generator=generate_role
):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description="Interactive local applicant reviewer")
    parser.add_argument("applicants", nargs="?", default=str(SAMPLE))
    parser.add_argument("--role", help="Load a saved role and evaluate at startup")
    parser.add_argument("--demo", action="store_true", help="Start with the included sample role")
    parser.add_argument("--model", default=os.environ.get("APPLICANT_MODEL", "llama3.2:3b"))
    parser.add_argument(
        "--ollama-url", default=os.environ.get("APPLICANT_OLLAMA_URL", "http://127.0.0.1:11434")
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use cached embedding/evidence models"
    )
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--session-dir", default="output/applicant-session")
    parser.add_argument(
        "--plain", action="store_true", help="Disable terminal colors and animations"
    )
    args = parser.parse_args(argv)
    console = console or Console(
        no_color=args.plain or "NO_COLOR" in os.environ,
        force_terminal=False if args.plain else None,
        highlight=False,
    )
    try:
        app = Reviewer(
            console,
            SessionStore(args.session_dir),
            model=args.model,
            url=args.ollama_url,
            offline=args.offline,
            budget=args.budget,
            model_factory=model_factory,
            generator=generator,
        )
        app.say("\napplicant reviewer", "bold cyan")
        app.say("Local matching · evidence you can inspect · your final judgment\n", "dim")
        app.load(args.applicants)
        if Path(args.applicants).resolve() == SAMPLE.resolve():
            app.say(
                "Using fictional sample applicants. 'load PATH' selects your own records.", "yellow"
            )
        if args.role or args.demo:
            app.use_role(read_json(args.role or DEMO_ROLE))
            app.run()
        else:
            app.say("Type a job description, 'demo' to try it now, or 'help'.", "cyan")
    except KeyboardInterrupt:
        console.print("\nCancelled.")
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(Text("Could not start: " + safe_text(exc), style="red"))
        return 2
    while True:
        try:
            prompt = "\nreview> " if app.role else "\njob> "
            raw = input_fn(prompt)
            if not app.command(raw, input_fn):
                break
        except EOFError:
            break
        except KeyboardInterrupt:
            app.say("\nCancelled. Type 'quit' to exit.", "yellow")
        except requests.ConnectionError:
            app.say(
                f"Couldn't reach local Ollama at {app.url}. Start 'ollama serve' and pull '{app.model_name}'. 'demo' works without Ollama.",
                "red",
            )
        except requests.Timeout:
            app.say(
                "The local generator timed out. Try again or use a saved role with 'role PATH'.",
                "red",
            )
        except (ValueError, OSError, RuntimeError) as exc:
            app.say(f"Could not complete that command: {exc}", "red")
    app.say("\nDone. Your saved review notes will be here next time.", "dim")
    return 0
