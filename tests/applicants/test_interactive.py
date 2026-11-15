import copy
import io
import json
from unittest.mock import Mock

import pytest
import requests
from rich.console import Console

from applicant_eval.interactive import Reviewer, main, one_path, safe_text
from applicant_eval.io import read_json, save
from applicant_eval.schema import Applicant, InputError, Role
from applicant_eval.session import SessionStore, load_applicants


@pytest.fixture
def cohort():
    return [
        {"id": "one", "evidence": [{"id": "e", "text": "I has_api has_sql has_production"}]},
        {"id": "two", "evidence": []},
    ]


@pytest.fixture
def reviewer(tmp_path, role_data, models, cohort):
    output = io.StringIO()
    app = Reviewer(
        Console(file=output, force_terminal=False, width=100),
        SessionStore(tmp_path / "session"),
        model_factory=lambda **kwargs: models,
        generator=lambda *args: copy.deepcopy(role_data),
    )
    app.applicants = [Applicant.from_dict(r) for r in cohort]
    app.output = output
    return app


def test_job_runs_and_reuses_models_and_profiles(reviewer, role_data):
    reviewer.generator = Mock(return_value=role_data)
    reviewer.model_factory = Mock(wraps=reviewer.model_factory)
    reviewer.job("Backend role")
    reviewer.job("Backend role")
    assert reviewer.generator.call_count == 1
    assert reviewer.model_factory.call_count == 1
    assert "Using cached profiles" in reviewer.output.getvalue()
    assert len(reviewer.report["results"]) == 2


def test_fresh_regenerates(reviewer, role_data):
    reviewer.generator = Mock(return_value=role_data)
    reviewer.job("Backend role")
    reviewer.command("fresh")
    assert reviewer.generator.call_count == 2


def test_invalid_generation_preserves_previous_results(reviewer):
    reviewer.job("First role")
    previous = copy.deepcopy(reviewer.report)
    reviewer.generator = lambda *args: {}
    with pytest.raises(InputError):
        reviewer.job("New role")
    assert reviewer.report == previous


def test_failed_load_preserves_current_cohort_and_results(reviewer, tmp_path):
    reviewer.job("Backend role")
    before = copy.deepcopy(reviewer.report)
    path = tmp_path / "bad.jsonl"
    path.write_text("not json")
    with pytest.raises(InputError):
        reviewer.load(path)
    assert reviewer.report == before
    assert len(reviewer.applicants) == 2


def test_mark_persists_without_changing_model_assessment(reviewer):
    reviewer.job("Backend role")
    before = copy.deepcopy(reviewer.report["results"][0])
    reviewer.command("mark 1 shortlist I'd like to discuss this project")
    result = reviewer.report["results"][0]
    assert result["human_review"]["note"] == "I'd like to discuss this project"
    assert {k: v for k, v in result.items() if k != "human_review"} == before
    reviewer.run()
    assert reviewer.report["results"][0]["human_review"]["status"] == "shortlist"
    reviewer.command("clear 1")
    assert "human_review" not in reviewer.report["results"][0]


def test_review_does_not_leak_to_changed_resume(reviewer):
    reviewer.job("Backend role")
    reviewer.mark("1", "pass", "Needs review")
    reviewer.applicants = [Applicant.from_dict({"id": "one", "evidence": []})]
    reviewer.run()
    assert "human_review" not in reviewer.report["results"][0]


def test_review_does_not_leak_to_different_role(reviewer, role_data):
    reviewer.job("Backend role")
    reviewer.mark("1", "hold")
    changed = copy.deepcopy(role_data)
    changed["criteria"][0]["hypothesis"] = "a different capability"
    reviewer.use_role(changed)
    reviewer.run()
    assert all("human_review" not in r for r in reviewer.report["results"])


def test_saved_review_reloads_in_new_session(reviewer):
    reviewer.job("Backend role")
    reviewer.mark("1", "shortlist")
    other = SessionStore(reviewer.store.directory)
    assert other.reviews(reviewer.role, reviewer.applicants)["one"]["status"] == "shortlist"


def test_corrupt_review_file_is_preserved(reviewer):
    reviewer.job("Backend role")
    path = reviewer.store.review_path(reviewer.role)
    save(path, '{"version":1,"entries":{"one":{"status":"invented"}}}')
    before = path.read_bytes()
    with pytest.raises(InputError):
        reviewer.mark("1", "hold")
    assert path.read_bytes() == before


def test_export_includes_human_notes_and_escapes_html(reviewer, tmp_path):
    reviewer.job("Backend role")
    reviewer.mark("1", "hold", "<script>alert('x')</script>")
    directory = tmp_path / "export with spaces"
    reviewer.export(f'"{directory}"')
    data = read_json(directory / "applicants.json")
    assert data["results"][0]["human_review"]["status"] == "hold"
    html = (directory / "applicants.html").read_text()
    assert "Your review:" in html and "<script>" not in html
    assert "&lt;script&gt;" in html
    Role.from_dict(read_json(directory / "role.json"))


def test_budget_and_queue(reviewer):
    reviewer.job("Backend role")
    reviewer.command("budget 1")
    assert len(reviewer.report["review_queue"]) == 1
    reviewer.command("queue")
    with pytest.raises(InputError):
        reviewer.command("budget -1")
    assert reviewer.budget == 1


@pytest.mark.parametrize(
    "command", ["1", "run", "profiles", "queue", "export", "fresh", "mark 1 hold", "clear 1"]
)
def test_commands_without_role_fail_helpfully(reviewer, command):
    with pytest.raises(InputError):
        reviewer.command(command)


@pytest.mark.parametrize(
    "command", ["0", "inspect 99", "show abc", "mark", "mark 1 unknown", "load", "load one two"]
)
def test_bad_commands_do_not_destroy_results(reviewer, command):
    reviewer.job("Backend role")
    before = copy.deepcopy(reviewer.report)
    with pytest.raises((InputError, ValueError)):
        reviewer.command(command)
    assert reviewer.report == before


def test_inspection_shows_evidence(reviewer):
    reviewer.job("Backend role")
    reviewer.command("1")
    output = reviewer.output.getvalue()
    assert "I has_api has_sql has_production" in output
    assert "Source e, characters" in output
    assert "Model support" in output


def test_multiline_job(reviewer):
    lines = iter(["Build APIs.", "Write SQL.", "."])
    reviewer.command("paste", input_fn=lambda _: next(lines))
    assert reviewer.description == "Build APIs.\nWrite SQL."


def test_terminal_control_characters_are_inert(reviewer, role_data):
    assert "\x1b" not in safe_text("x\x1b[2Jy\u202ez")
    assert "\u202e" not in safe_text("x\u202ey")
    role_data["title"] = "[bold]literal[/bold]\x1b[2J"
    reviewer.use_role(role_data)
    reviewer.run()
    output = reviewer.output.getvalue()
    assert "[bold]literal[/bold]" in output
    assert "\x1b" not in output


def test_quit_aliases_and_blank_input(reviewer):
    assert reviewer.command("")
    for command in ["quit", "q", "exit"]:
        assert reviewer.command(command) is False


def test_folder_loader_preserves_text(tmp_path):
    (tmp_path / "resume one.txt").write_text("  I built software.\n")
    (tmp_path / "resume two.md").write_text("I wrote tests.")
    (tmp_path / ".hidden.txt").write_text("not loaded")
    (tmp_path / "irrelevant.pdf").write_text("not parsed")
    values = load_applicants(tmp_path)
    assert len(values) == 2
    assert values[0].evidence[0].text == "  I built software.\n"


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_json_loader(tmp_path, cohort, suffix):
    path = tmp_path / ("applicants" + suffix)
    path.write_text(json.dumps(cohort) if suffix == ".json" else "\n".join(map(json.dumps, cohort)))
    assert len(load_applicants(path)) == 2


def test_single_text_loader(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("A resume")
    assert load_applicants(path)[0].id == "resume.txt"


def test_duplicate_ids_rejected(tmp_path, cohort):
    path = tmp_path / "x.json"
    path.write_text(json.dumps([cohort[0], cohort[0]]))
    with pytest.raises(InputError, match="duplicate"):
        load_applicants(path)


@pytest.mark.parametrize("value", [[], {}, 42])
def test_empty_or_malformed_input_rejected(tmp_path, value):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(value))
    with pytest.raises(InputError):
        load_applicants(path)


def test_no_text_in_directory_rejected(tmp_path):
    with pytest.raises(InputError, match="No .txt"):
        load_applicants(tmp_path)


def test_unsupported_format_explained(tmp_path):
    with pytest.raises(InputError, match="PDF"):
        load_applicants(tmp_path / "resume.pdf")


def test_path_quotes():
    assert str(one_path('"folder with spaces"')) == "folder with spaces"


def test_terminal_loop_recovers_from_generator_error(tmp_path, cohort, models):
    path = tmp_path / "applicants.json"
    path.write_text(json.dumps(cohort))
    output = io.StringIO()
    commands = iter(["A backend job", "help", "quit"])
    broken = Mock(side_effect=requests.ConnectionError("offline"))
    assert (
        main(
            [str(path), "--session-dir", str(tmp_path / "session"), "--plain"],
            input_fn=lambda _: next(commands),
            console=Console(file=output, force_terminal=False),
            model_factory=lambda **kw: models,
            generator=broken,
        )
        == 0
    )
    assert "Couldn't reach local Ollama" in output.getvalue()
    assert "Done." in output.getvalue()


def test_startup_bad_path_is_clear(tmp_path, models):
    output = io.StringIO()
    result = main(
        [str(tmp_path / "missing.jsonl")],
        console=Console(file=output, force_terminal=False),
        model_factory=lambda **kw: models,
    )
    assert result == 2
    assert "Could not start" in output.getvalue()


def test_eof_exits_cleanly(tmp_path, cohort):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(cohort))

    def end(_):
        raise EOFError

    assert (
        main(
            [str(path), "--session-dir", str(tmp_path / "s")],
            input_fn=end,
            console=Console(file=io.StringIO()),
        )
        == 0
    )


def test_module_no_args_opens_interactive(monkeypatch):
    from applicant_eval.__main__ import main as entry

    run = Mock(return_value=0)
    monkeypatch.setattr("applicant_eval.interactive.main", run)
    assert entry([]) == 0
    run.assert_called_once_with([])


def test_module_interactive_forwards_arguments(monkeypatch):
    from applicant_eval.__main__ import main as entry

    run = Mock(return_value=0)
    monkeypatch.setattr("applicant_eval.interactive.main", run)
    assert entry(["interactive", "--demo"]) == 0
    run.assert_called_once_with(["--demo"])
