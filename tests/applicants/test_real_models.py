import json
from pathlib import Path

import numpy as np
import pytest

from applicant_eval.engine import evaluate
from applicant_eval.schema import Applicant, Role

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.model


def test_real_embeddings_are_normalized(real_models):
    v = real_models.encode(["I built a web API.", "I baked a cake."])
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    assert float(v[0] @ v[1]) < 0.7


def test_real_long_input_tail_and_offsets(real_models):
    text = "I arranged meetings and ordered stationery. " * 300 + "I built an API."
    spans = list(real_models.spans(text))
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert any("I built an API." in text[a:b] for a, b in spans)
    assert all(not text[spans[i][1] : spans[i + 1][0]].strip() for i in range(len(spans) - 1))
    for a, b in spans:
        assert (
            len(real_models.encoder.tokenizer(text[a:b])["input_ids"])
            <= real_models.encoder.max_seq_length
        )


def test_real_negation_and_positive_pair(real_models):
    positive, negative = real_models.verify(
        [
            ("I built a REST API.", "This person has built an API."),
            ("I have never built an API.", "This person has built an API."),
        ]
    )
    assert positive[1] > 0.8
    assert negative[0] > 0.8


def test_real_development_cases_and_source_integrity(real_models):
    role = Role.from_dict(json.loads((ROOT / "examples/applicants/backend_role.json").read_text()))
    records = [
        json.loads(line) for line in (ROOT / "benchmarks/applicants.jsonl").read_text().splitlines()
    ]
    records = [r for r in records if r["split"] == "dev"]
    report = evaluate(role, [Applicant.from_dict(r) for r in records], real_models)
    outputs = {r["applicant_id"]: r for r in report["results"]}
    for identifier in (
        "dev-recruiter",
        "dev-negated",
        "dev-future",
        "dev-no-tests",
        "dev-no-sql",
        "dev-empty",
        "dev-conflict",
    ):
        assert outputs[identifier]["review_status"] == "needs_review"
    sources = {r["id"]: {e["id"]: e["text"] for e in r["evidence"]} for r in records}
    for r in report["results"]:
        for c in r["evidence"]:
            for s in c["sources"]:
                assert (
                    sources[r["applicant_id"]][s["evidence_id"]][s["start"] : s["end"]]
                    == s["quote"]
                )
