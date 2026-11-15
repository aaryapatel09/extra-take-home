"""Validated inputs. Identity metadata and benchmark labels never enter scoring."""

import math
from dataclasses import dataclass


class InputError(ValueError):
    pass


def text(value, field, limit=10000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise InputError(f"{field} must be nonempty text of at most {limit} characters")
    return value.strip()


def objects(value, field, minimum=1, maximum=100):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise InputError(f"{field} must contain {minimum}–{maximum} entries")
    return value


def unique(values, field):
    if len(values) != len(set(values)):
        raise InputError(f"duplicate {field}")


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str
    hypothesis: str
    required: bool


@dataclass(frozen=True)
class Profile:
    id: str
    description: str
    criteria: tuple[str, ...]
    examples: tuple[str, ...]
    near_misses: tuple[str, ...]


@dataclass(frozen=True)
class Role:
    id: str
    title: str
    description: str
    criteria: tuple[Criterion, ...]
    profiles: tuple[Profile, ...]

    @classmethod
    def from_dict(cls, data):
        try:
            criteria = []
            for c in objects(data["criteria"], "criteria", maximum=20):
                required = c.get("required", False)
                if not isinstance(required, bool):
                    raise InputError("criterion.required must be a boolean")
                criteria.append(
                    Criterion(
                        text(c["id"], "criterion.id", 80),
                        text(c["description"], "criterion.description", 600),
                        text(c["hypothesis"], "criterion.hypothesis", 300),
                        required,
                    )
                )
            unique([c.id for c in criteria], "criterion IDs")
            if not any(c.required for c in criteria):
                raise InputError("at least one essential criterion is required")
            profiles = []
            for p in objects(data["profiles"], "profiles", maximum=40):
                ids = tuple(
                    text(x, "profile.criteria", 80)
                    for x in objects(p["criteria"], "profile.criteria", maximum=20)
                )
                unique(ids, "profile criterion IDs")
                if not set(ids) <= {c.id for c in criteria}:
                    raise InputError("profile references an unknown criterion")
                examples = tuple(
                    text(x, "profile.example", 2000)
                    for x in objects(p["examples"], "profile.examples", 2, 20)
                )
                unique(examples, "examples within a profile")
                misses = tuple(
                    text(x, "near_miss", 2000)
                    for x in objects(p.get("near_misses", []), "near_misses", 0, 20)
                )
                profiles.append(
                    Profile(
                        text(p["id"], "profile.id", 80),
                        text(p["description"], "profile.description", 600),
                        ids,
                        examples,
                        misses,
                    )
                )
            unique([p.id for p in profiles], "profile IDs")
            return cls(
                text(data["id"], "role.id", 80),
                text(data["title"], "role.title", 200),
                text(data["description"], "role.description", 5000),
                tuple(criteria),
                tuple(profiles),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise InputError(f"invalid role structure: {exc}") from exc


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str


@dataclass(frozen=True)
class Applicant:
    id: str
    evidence: tuple[Evidence, ...]

    @classmethod
    def from_dict(cls, data):
        try:
            # All other fields (name, school, age, benchmark labels, etc.) are ignored.
            evidence = tuple(
                Evidence(
                    text(e["id"], "evidence.id", 80),
                    e["text"] if text(e["text"], "evidence.text", 50000) else "",
                )
                for e in objects(data["evidence"], "evidence", 0, 100)
            )
            unique([e.id for e in evidence], "evidence IDs")
            if sum(len(e.text) for e in evidence) > 100000:
                raise InputError("applicant evidence exceeds 100000 characters")
            return cls(text(data["id"], "applicant.id", 100), evidence)
        except (KeyError, TypeError, AttributeError) as exc:
            raise InputError(f"invalid applicant structure: {exc}") from exc


@dataclass(frozen=True)
class Settings:
    evidence_threshold: float = 0.8
    contradiction_threshold: float = 0.8
    evidence_top_k: int = 3
    min_evidence_similarity: float = 0.25

    def __post_init__(self):
        for name in ("evidence_threshold", "contradiction_threshold"):
            v = getattr(self, name)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0.5 < v <= 1
            ):
                raise InputError(f"{name} must be finite and in (0.5, 1]")
        if type(self.evidence_top_k) is not int or not 1 <= self.evidence_top_k <= 20:
            raise InputError("evidence_top_k must be an integer in [1, 20]")
        if (
            not math.isfinite(self.min_evidence_similarity)
            or not -1 <= self.min_evidence_similarity <= 1
        ):
            raise InputError("min_evidence_similarity must be finite and in [-1, 1]")
