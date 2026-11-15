"""Local profile cache and human review notes, scoped to exact input contents."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .io import read_json, read_jsonl, save
from .schema import Applicant, InputError, Role, unique

REVIEW_STATES = {"shortlist", "hold", "pass"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def load_applicants(path):
    path = Path(path).expanduser()
    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in {".txt", ".md"} and not p.name.startswith(".")
        )
        if not files:
            raise InputError(
                "No .txt or .md resumes found. Use one applicant per file, or a JSONL file."
            )
        records = [
            {"id": p.name, "evidence": [{"id": p.name, "text": p.read_text(encoding="utf-8")}]}
            for p in files
        ]
    elif path.suffix.lower() in {".txt", ".md"}:
        records = [
            {
                "id": path.name,
                "evidence": [{"id": path.name, "text": path.read_text(encoding="utf-8")}],
            }
        ]
    elif path.suffix.lower() == ".json":
        records = read_json(path)
        if not isinstance(records, list):
            raise InputError("Applicant JSON must contain an array of records.")
    elif path.suffix.lower() == ".jsonl":
        records = read_jsonl(path)
    else:
        raise InputError(
            "Use a .jsonl file, a JSON array, or a folder of .txt/.md resumes. PDF parsing is not supported."
        )
    applicants = [Applicant.from_dict(r) for r in records]
    unique([a.id for a in applicants], "applicant IDs")
    if not applicants:
        raise InputError("No applicants found in this file.")
    return applicants


class SessionStore:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser()

    def cache_path(self, description, model):
        return (
            self.directory
            / "roles"
            / (digest({"version": 4, "description": description.strip(), "model": model}) + ".json")
        )

    def cached_role(self, description, model):
        path = self.cache_path(description, model)
        if not path.exists():
            return None
        data = read_json(path)
        Role.from_dict(data)
        return data

    def save_role(self, description, model, data):
        Role.from_dict(data)
        save(
            self.cache_path(description, model),
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        )

    def review_path(self, role):
        return self.directory / "reviews" / (digest(asdict(role)) + ".json")

    def _read_reviews(self, role):
        path = self.review_path(role)
        if not path.exists():
            return {}
        data = read_json(path)
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("entries"), dict)
        ):
            raise InputError(
                f"Invalid review file: {path}. It has been preserved; move it aside to start fresh."
            )
        for value in data["entries"].values():
            if (
                not isinstance(value, dict)
                or value.get("status") not in REVIEW_STATES
                or not isinstance(value.get("note"), str)
                or not isinstance(value.get("applicant_digest"), str)
            ):
                raise InputError(f"Invalid review entry in {path}; the file has been preserved.")
        return data["entries"]

    def reviews(self, role, applicants):
        entries = self._read_reviews(role)
        return {
            a.id: entries[a.id]
            for a in applicants
            if a.id in entries and entries[a.id]["applicant_digest"] == digest(asdict(a))
        }

    def mark(self, role, applicant, status, note=""):
        if status not in REVIEW_STATES:
            raise InputError("Review status must be shortlist, hold, or pass.")
        if len(note) > 4000:
            raise InputError("Review note is too long (maximum 4000 characters).")
        entries = self._read_reviews(role)
        entries[applicant.id] = {
            "status": status,
            "note": note,
            "applicant_digest": digest(asdict(applicant)),
        }
        save(
            self.review_path(role),
            json.dumps({"version": 1, "entries": entries}, indent=2, ensure_ascii=False) + "\n",
        )

    def clear(self, role, applicant):
        entries = self._read_reviews(role)
        entries.pop(applicant.id, None)
        save(
            self.review_path(role),
            json.dumps({"version": 1, "entries": entries}, indent=2, ensure_ascii=False) + "\n",
        )
