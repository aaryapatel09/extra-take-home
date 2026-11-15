import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from applicant_eval.__main__ import main, read_jsonl, save
from applicant_eval.generation import PROFILE_ROUTES, generate_role
from applicant_eval.schema import InputError

JOB = "Build an API and write SQL; production and research work are alternatives."


@pytest.fixture
def draft_data(role_data):
    draft = deepcopy(role_data)
    draft.pop("profiles")
    for criterion in draft["criteria"]:
        criterion["required"] = criterion["id"] in {"api", "sql"}
        criterion["source_quote"] = {"api": "API", "sql": "SQL"}.get(
            criterion["id"], criterion["id"]
        )
        criterion["hypothesis"] = "This person has completed " + criterion["id"] + " work."
    return draft


@pytest.fixture
def examples():
    return {
        path: [
            {
                "api": f"I built an API for project {i}.",
                "sql": f"I wrote SQL queries for project {i}.",
            }
            for i in range(3)
        ]
        for path in PROFILE_ROUTES
    }


def responses(monkeypatch, *payloads):
    sequence = []
    for payload in payloads:
        response = Mock()
        response.json.return_value = {"response": json.dumps(payload)}
        sequence.append(response)
    post = Mock(side_effect=sequence)
    monkeypatch.setattr("applicant_eval.generation.requests.post", post)
    return post


def test_generator_validates_and_records_origin(monkeypatch, draft_data, examples):
    post = responses(monkeypatch, draft_data, examples)
    result = generate_role(JOB, model="local-test")
    assert result["provenance"]["model"] == "local-test"
    assert result["provenance"]["job_description"] == JOB
    assert result["provenance"]["generation_attempts"] == 2
    assert post.call_count == 2
    assert post.call_args.kwargs["json"]["format"]["type"] == "object"
    assert post.call_args.kwargs["json"]["keep_alive"] == 0
    assert post.call_args.kwargs["timeout"] == 180


def test_source_quotes_preserve_original_job_letter_case(monkeypatch, draft_data, examples):
    draft_data["criteria"][0]["source_quote"] = "api"
    responses(monkeypatch, draft_data, examples)
    result = generate_role(JOB)
    assert result["criteria"][0]["source_quote"] == "API"


def test_generator_repairs_invalid_first_response(monkeypatch, draft_data, examples):
    post = responses(monkeypatch, {}, draft_data, examples)
    result = generate_role(JOB)
    assert result["provenance"]["criterion_attempts"] == 2
    assert post.call_count == 3
    assert "failed validation" in post.call_args_list[1].kwargs["json"]["prompt"]


def test_generated_profiles_cover_every_essential(monkeypatch, draft_data, examples):
    responses(monkeypatch, draft_data, examples)
    result = generate_role(JOB)
    assert [p["id"] for p in result["profiles"]] == list(PROFILE_ROUTES)
    for profile in result["profiles"]:
        assert profile["criteria"] == ["api", "sql"]
        assert len(profile["examples"]) == 3
        assert all("built an API" in e and "wrote SQL" in e for e in profile["examples"])
        assert profile["near_misses"] == []


def test_invented_tenure_is_repaired(monkeypatch, draft_data, examples):
    bad = deepcopy(examples)
    bad["direct-experience"][0]["api"] = "I have 3+ years of experience building APIs."
    post = responses(monkeypatch, draft_data, bad, examples)
    result = generate_role(JOB)
    assert result["provenance"]["example_attempts"] == 2
    assert "invented years" in post.call_args.kwargs["json"]["prompt"]


@pytest.mark.parametrize("mutation", ["invented_quote", "bad_hypothesis", "duplicate_id"])
def test_invalid_criteria_are_repaired(monkeypatch, draft_data, examples, mutation):
    bad = deepcopy(draft_data)
    if mutation == "invented_quote":
        bad["criteria"][0]["source_quote"] = "Invented management responsibility"
    elif mutation == "bad_hypothesis":
        bad["criteria"][0]["hypothesis"] = "Build an API"
    else:
        bad["criteria"][1]["id"] = bad["criteria"][0]["id"]
    post = responses(monkeypatch, bad, draft_data, examples)
    result = generate_role(JOB)
    assert result["provenance"]["criterion_attempts"] == 2
    assert post.call_count == 3


@pytest.mark.parametrize(
    "mutation", ["missing_skill", "missing_path", "short_path", "duplicate_example"]
)
def test_incomplete_profiles_are_repaired(monkeypatch, draft_data, examples, mutation):
    bad = deepcopy(examples)
    if mutation == "missing_skill":
        bad["direct-experience"][0].pop("sql")
    elif mutation == "missing_path":
        bad.pop("direct-experience")
    elif mutation == "short_path":
        bad["direct-experience"].pop()
    else:
        bad["direct-experience"][1] = bad["direct-experience"][0]
    post = responses(monkeypatch, draft_data, bad, examples)
    result = generate_role(JOB)
    assert result["provenance"]["example_attempts"] == 2
    assert post.call_count == 3


def test_example_repairs_are_bounded(monkeypatch, draft_data):
    post = responses(monkeypatch, draft_data, {}, {})
    with pytest.raises(InputError, match="after two attempts"):
        generate_role(JOB)
    assert post.call_count == 3


@pytest.mark.parametrize("raw", ["not json", "{}", "[]", '{"criteria":null}'])
def test_generator_rejects_malformed_role(monkeypatch, raw):
    response = Mock()
    response.json.return_value = {"response": raw}
    monkeypatch.setattr("applicant_eval.generation.requests.post", Mock(return_value=response))
    with pytest.raises(InputError):
        generate_role("Build an API")


def test_http_failure_is_not_hidden(monkeypatch):
    import requests

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("server error")
    monkeypatch.setattr("applicant_eval.generation.requests.post", Mock(return_value=response))
    with pytest.raises(requests.HTTPError):
        generate_role("Build an API")


def test_bad_jsonl_reports_line(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("{}\n\ninvalid\n")
    with pytest.raises(InputError, match="line 3"):
        read_jsonl(p)


def test_atomic_save_creates_parent_and_replaces(tmp_path):
    p = tmp_path / "nested" / "report.json"
    save(p, "first")
    save(p, "second")
    assert p.read_text() == "second"
    assert len(list(p.parent.iterdir())) == 1


def test_cli_validates_before_loading_model(tmp_path, capsys):
    role = tmp_path / "role.json"
    role.write_text("{}")
    assert (
        main(
            [
                "evaluate",
                "--role",
                str(role),
                "--applicants",
                "missing",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "invalid role" in capsys.readouterr().err


def test_cli_evaluation_writes_reports(tmp_path, monkeypatch, role_data, models):
    role = tmp_path / "role.json"
    role.write_text(json.dumps(role_data))
    applicants = tmp_path / "applicants.jsonl"
    applicants.write_text(json.dumps({"id": "empty", "evidence": []}) + "\n")
    monkeypatch.setattr("applicant_eval.models.LocalModels", lambda **kwargs: models)
    output, html = tmp_path / "out.json", tmp_path / "out.html"
    assert (
        main(
            [
                "evaluate",
                "--role",
                str(role),
                "--applicants",
                str(applicants),
                "--output",
                str(output),
                "--html",
                str(html),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["results"][0]["review_status"] == "needs_review"
    assert html.read_text().startswith("<!doctype html>")


def test_cli_benchmark(tmp_path, monkeypatch, role_data, models):
    role = tmp_path / "role.json"
    role.write_text(json.dumps(role_data))
    applicants = tmp_path / "data.jsonl"
    applicants.write_text(
        json.dumps({"id": "a", "evidence": [], "split": "test", "expected_qualified": False})
    )
    monkeypatch.setattr("applicant_eval.models.LocalModels", lambda **kwargs: models)
    output = tmp_path / "benchmark.json"
    assert (
        main(
            [
                "benchmark",
                "--role",
                str(role),
                "--applicants",
                str(applicants),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["independent_records"] == 1


def test_cli_generator(tmp_path, monkeypatch, role_data):
    description = tmp_path / "job.txt"
    description.write_text("Implement APIs")
    output = tmp_path / "role.json"
    monkeypatch.setattr("applicant_eval.generation.generate_role", lambda *args: role_data)
    assert main(["generate-role", "--description", str(description), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["id"] == "role"
