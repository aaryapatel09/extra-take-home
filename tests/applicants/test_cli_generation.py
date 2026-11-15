import json
from unittest.mock import Mock

import pytest

from applicant_eval.__main__ import main, read_jsonl, save
from applicant_eval.generation import generate_role
from applicant_eval.schema import InputError


def test_generator_validates_and_records_origin(monkeypatch, role_data):
    response = Mock()
    response.json.return_value = {"response": json.dumps(role_data)}
    post = Mock(return_value=response)
    monkeypatch.setattr("applicant_eval.generation.requests.post", post)
    result = generate_role("Build an API", model="local-test")
    assert result["provenance"]["model"] == "local-test"
    assert post.call_args.kwargs["json"]["format"] == "json"
    assert post.call_args.kwargs["timeout"] == 180


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
