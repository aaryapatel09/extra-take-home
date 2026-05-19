#!/usr/bin/env python3
"""tests for classifier.py"""

import json
import base64
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from classifier import (
    decode_body,
    email_to_text,
    parse_json_response,
    _parse_examples,
    cosine_sim,
    load_messages,
    generate_synthetic_examples,
    generate_explanations_batch,
    get_feedback_path,
    load_feedback,
    save_feedback,
    classify,
    classify_multi,
    MARGIN_THRESHOLD,
)


# ── fixtures ───────────────────────────────────────────────

SAMPLE_MSG = {
    "id": "174a9",
    "subject": "Your Delta eTicket Receipt",
    "from": "Delta <no-reply@delta.com>",
    "to": ["sam@example.com"],
    "snippet": "Thanks for flying with us",
    "body": base64.b64encode(b"Flight DL1234 from JFK to LAX. Total: $342.50").decode(),
    "date": "2025-08-12T14:33:22Z",
}

SAMPLE_MSGS = [
    SAMPLE_MSG,
    {
        "id": "174b2",
        "subject": "Weekly AI Research Digest",
        "from": "AI Newsletter <digest@ainews.io>",
        "to": ["sam@example.com"],
        "snippet": "This week in AI",
        "body": base64.b64encode(b"New transformer architectures reduce compute by 40%").decode(),
        "date": "2025-08-11T09:15:00Z",
    },
    {
        "id": "174c5",
        "subject": "Re: Thanksgiving plans",
        "from": "Mom <mom@gmail.com>",
        "to": ["sam@example.com"],
        "snippet": "Hi sweetie",
        "body": base64.b64encode(b"Are you coming home for Thanksgiving?").decode(),
        "date": "2025-08-10T18:22:11Z",
    },
]

FAKE_SYNTHETIC = [
    {"label": "yes", "from": "Delta <noreply@delta.com>", "subject": "eTicket Receipt", "snippet": "", "body": "Flight confirmed total charged"},
    {"label": "yes", "from": "United <noreply@united.com>", "subject": "Booking Confirmation", "snippet": "", "body": "Business travel SFO to ORD"},
    {"label": "yes", "from": "AA <receipts@aa.com>", "subject": "Trip Receipt", "snippet": "", "body": "Your flight receipt attached"},
    {"label": "yes", "from": "Southwest <fly@southwest.com>", "subject": "Travel Itinerary", "snippet": "", "body": "Work trip LAX to DEN"},
    {"label": "yes", "from": "JetBlue <noreply@jetblue.com>", "subject": "Your Receipt", "snippet": "", "body": "Booking receipt for BOS flight"},
    {"label": "no", "from": "Amazon <ship@amazon.com>", "subject": "Order Shipped", "snippet": "", "body": "Your USB hub is on the way"},
    {"label": "no", "from": "Mom <mom@gmail.com>", "subject": "Thanksgiving", "snippet": "", "body": "Are you coming home"},
    {"label": "no", "from": "Calendar <cal@acme.com>", "subject": "Q3 Meeting", "snippet": "", "body": "Team standup at 2pm"},
    {"label": "no", "from": "PWL <news@paperswelove.org>", "subject": "August Papers", "snippet": "", "body": "Featured paper on attention"},
    {"label": "no", "from": "GitHub <noreply@github.com>", "subject": "PR merged", "snippet": "", "body": "Your pull request was merged"},
    {"label": "edge", "from": "Expedia <trips@expedia.com>", "subject": "Hotel Booking", "snippet": "", "body": "Hotel reservation confirmed"},
    {"label": "edge", "from": "Uber <receipts@uber.com>", "subject": "Trip Receipt", "snippet": "", "body": "Your ride receipt from Tuesday"},
]

FAKE_EXPLANATIONS = json.dumps([
    "Delta eTicket is a travel receipt from an airline",
    "AI newsletter is not a travel receipt",
    "Mom's email is family communication, not travel",
])


# ── decode_body ────────────────────────────────────────────

def test_decode_body_plain_text():
    encoded = base64.b64encode(b"hello world").decode()
    assert decode_body(encoded) == "hello world"


def test_decode_body_strips_html():
    html = "<html><body><p>hello</p><br><b>world</b></body></html>"
    encoded = base64.b64encode(html.encode()).decode()
    result = decode_body(encoded)
    assert "<" not in result
    assert "hello" in result
    assert "world" in result


def test_decode_body_nested_html():
    html = "<div><ul><li>item 1</li><li>item 2</li></ul></div>"
    encoded = base64.b64encode(html.encode()).decode()
    result = decode_body(encoded)
    assert "<" not in result
    assert "item 1" in result
    assert "item 2" in result


def test_decode_body_collapses_whitespace():
    text = "hello     world\n\n\nnewlines"
    encoded = base64.b64encode(text.encode()).decode()
    assert decode_body(encoded) == "hello world newlines"


def test_decode_body_empty():
    encoded = base64.b64encode(b"").decode()
    assert decode_body(encoded) == ""


def test_decode_body_malformed():
    result = decode_body("not-valid-base64!!!")
    assert "[decode error" in result


def test_decode_body_unicode():
    text = "Confirmación de vuelo — gracias"
    encoded = base64.b64encode(text.encode("utf-8")).decode()
    result = decode_body(encoded)
    assert "Confirmación" in result
    assert "gracias" in result


def test_decode_body_html_entities_mixed():
    html = "<p>Price: $100 &amp; tax</p>"
    encoded = base64.b64encode(html.encode()).decode()
    result = decode_body(encoded)
    assert "<" not in result
    assert "$100" in result


# ── email_to_text ──────────────────────────────────────────

def test_email_to_text_all_fields():
    text = email_to_text(SAMPLE_MSG)
    assert "From: Delta <no-reply@delta.com>" in text
    assert "To: sam@example.com" in text
    assert "Date: 2025-08-12T14:33:22Z" in text
    assert "Subject: Your Delta eTicket Receipt" in text
    assert "Snippet: Thanks for flying with us" in text
    assert "Flight DL1234" in text


def test_email_to_text_missing_fields():
    text = email_to_text({})
    assert "From:" in text
    assert "Subject:" in text
    assert "To:" in text
    assert "Date:" in text


def test_email_to_text_string_to_field():
    msg = {"to": "single@example.com", "from": "", "subject": "", "snippet": "", "body": ""}
    text = email_to_text(msg)
    assert "To: single@example.com" in text


def test_email_to_text_multiple_recipients():
    msg = {
        "to": ["a@test.com", "b@test.com", "c@test.com"],
        "from": "", "subject": "", "snippet": "",
        "body": base64.b64encode(b"test").decode(),
    }
    text = email_to_text(msg)
    assert "a@test.com, b@test.com, c@test.com" in text


# ── parse_json_response ────────────────────────────────────

def test_parse_json_clean_array():
    assert parse_json_response('[{"a": 1}]') == [{"a": 1}]


def test_parse_json_clean_object():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_string_array():
    result = parse_json_response('["hello", "world"]')
    assert result == ["hello", "world"]


def test_parse_json_code_fenced():
    raw = '```json\n[{"a": 1}]\n```'
    assert parse_json_response(raw) == [{"a": 1}]


def test_parse_json_code_fenced_no_lang():
    raw = '```\n[{"a": 1}]\n```'
    assert parse_json_response(raw) == [{"a": 1}]


def test_parse_json_with_preamble():
    raw = 'Here are the results:\n[{"a": 1}, {"b": 2}]'
    assert parse_json_response(raw) == [{"a": 1}, {"b": 2}]


def test_parse_json_with_trailing_text():
    raw = '[{"a": 1}]\n\nHope this helps!'
    result = parse_json_response(raw)
    assert result == [{"a": 1}]


def test_parse_json_partial_recovery():
    raw = 'broken start {"a": 1} middle {"b": 2} end'
    result = parse_json_response(raw)
    assert len(result) == 2
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}


def test_parse_json_total_garbage():
    assert parse_json_response("not json at all lol") is None


def test_parse_json_empty():
    assert parse_json_response("") is None


def test_parse_json_whitespace_only():
    assert parse_json_response("   \n\n  ") is None


def test_parse_json_nested_objects():
    raw = '[{"outer": {"inner": "value"}}]'
    result = parse_json_response(raw)
    assert result[0]["outer"]["inner"] == "value"


# ── _parse_examples ────────────────────────────────────────

def test_parse_examples_splits_labels():
    examples = [
        {"label": "yes", "from": "a", "subject": "a", "snippet": "", "body": "a"},
        {"label": "no", "from": "b", "subject": "b", "snippet": "", "body": "b"},
        {"label": "edge", "from": "c", "subject": "c", "snippet": "", "body": "c"},
        {"label": "YES", "from": "d", "subject": "d", "snippet": "", "body": "d"},
    ]
    yes, no, edge = _parse_examples(examples)
    assert len(yes) == 2
    assert len(no) == 1
    assert len(edge) == 1


def test_parse_examples_skips_non_dicts():
    examples = [None, "string", 42, {"label": "yes", "from": "", "subject": "", "snippet": "", "body": ""}]
    yes, no, edge = _parse_examples(examples)
    assert len(yes) == 1
    assert len(no) == 0


def test_parse_examples_empty():
    yes, no, edge = _parse_examples([])
    assert yes == [] and no == [] and edge == []


def test_parse_examples_unknown_labels():
    examples = [
        {"label": "maybe", "from": "", "subject": "", "snippet": "", "body": ""},
        {"label": "", "from": "", "subject": "", "snippet": "", "body": ""},
    ]
    yes, no, edge = _parse_examples(examples)
    assert len(yes) == 0 and len(no) == 0 and len(edge) == 0


def test_parse_examples_text_format():
    examples = [{"label": "yes", "from": "sender@test.com", "subject": "Test", "snippet": "snip", "body": "content"}]
    yes, _, _ = _parse_examples(examples)
    assert "From: sender@test.com" in yes[0]
    assert "Subject: Test" in yes[0]
    assert "Body: content" in yes[0]


def test_parse_examples_whitespace_labels():
    examples = [
        {"label": " yes ", "from": "", "subject": "", "snippet": "", "body": ""},
        {"label": "  NO  ", "from": "", "subject": "", "snippet": "", "body": ""},
    ]
    yes, no, edge = _parse_examples(examples)
    assert len(yes) == 1
    assert len(no) == 1


def test_parse_examples_full_synthetic():
    yes, no, edge = _parse_examples(FAKE_SYNTHETIC)
    assert len(yes) == 5
    assert len(no) == 5
    assert len(edge) == 2


# ── cosine_sim ─────────────────────────────────────────────

def test_cosine_sim_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_sim(v, v) - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_sim(a, b)) < 1e-6


def test_cosine_sim_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_sim(a, b) - (-1.0)) < 1e-6


def test_cosine_sim_zero_vector():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_sim(a, b) == 0.0
    assert cosine_sim(b, a) == 0.0


def test_cosine_sim_both_zero():
    a = np.array([0.0, 0.0])
    assert cosine_sim(a, a) == 0.0


def test_cosine_sim_high_dimensional():
    rng = np.random.RandomState(42)
    a = rng.randn(384)
    b = rng.randn(384)
    result = cosine_sim(a, b)
    assert -1.0 <= result <= 1.0


def test_cosine_sim_scaled_vectors():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    assert abs(cosine_sim(a, b) - 1.0) < 1e-6


def test_cosine_sim_negative_values():
    a = np.array([-1.0, -2.0, -3.0])
    b = np.array([-1.0, -2.0, -3.0])
    assert abs(cosine_sim(a, b) - 1.0) < 1e-6


# ── load_messages ──────────────────────────────────────────

def test_load_messages(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "1", "subject": "test"}\n{"id": "2", "subject": "test2"}\n')
    msgs = load_messages(str(f))
    assert len(msgs) == 2
    assert msgs[0]["id"] == "1"


def test_load_messages_skips_blank_lines(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "1"}\n\n\n{"id": "2"}\n')
    msgs = load_messages(str(f))
    assert len(msgs) == 2


def test_load_messages_single(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "1"}\n')
    msgs = load_messages(str(f))
    assert len(msgs) == 1


def test_load_messages_preserves_all_fields(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(json.dumps(SAMPLE_MSG) + "\n")
    msgs = load_messages(str(f))
    assert msgs[0]["id"] == "174a9"
    assert msgs[0]["subject"] == "Your Delta eTicket Receipt"
    assert isinstance(msgs[0]["to"], list)


def test_load_real_messages():
    msgs = load_messages("messages.jsonl")
    assert len(msgs) == 12
    for msg in msgs:
        assert "id" in msg
        assert "subject" in msg
        assert "from" in msg
        assert "to" in msg
        assert "body" in msg
        assert "snippet" in msg


# ── feedback persistence ───────────────────────────────────

def test_get_feedback_path():
    path = get_feedback_path("work travel receipts")
    assert path == "feedback_work_travel_receipts.json"


def test_get_feedback_path_special_chars():
    path = get_feedback_path("AI/ML newsletters!!!")
    assert path == "feedback_ai_ml_newsletters.json"


def test_feedback_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    category = "test category"
    feedback = {"msg1": {"corrected_label": True}, "msg2": {"corrected_label": False}}
    save_feedback(category, feedback)
    loaded = load_feedback(category)
    assert loaded == feedback


def test_load_feedback_nonexistent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_feedback("nonexistent category") == {}


# ── generate_synthetic_examples ────────────────────────────

@patch("classifier.call_ollama")
def test_generate_synthetic_caches(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = json.dumps(FAKE_SYNTHETIC)
    yes, no, edge = generate_synthetic_examples("travel receipts", fresh=True)
    assert len(yes) == 5
    assert len(no) == 5
    assert len(edge) == 2
    assert mock_ollama.call_count == 1

    yes2, no2, edge2 = generate_synthetic_examples("travel receipts", fresh=False)
    assert len(yes2) == 5
    assert mock_ollama.call_count == 1  # used cache, no new call


@patch("classifier.call_ollama")
def test_generate_synthetic_fresh_bypasses_cache(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = json.dumps(FAKE_SYNTHETIC)
    generate_synthetic_examples("travel receipts", fresh=True)
    generate_synthetic_examples("travel receipts", fresh=True)
    assert mock_ollama.call_count == 2


@patch("classifier.call_ollama")
def test_generate_synthetic_bad_response(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = "not json at all"
    yes, no, edge = generate_synthetic_examples("bad category", fresh=True)
    assert yes == [] and no == [] and edge == []


@patch("classifier.call_ollama")
def test_generate_synthetic_uses_temp_zero(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = json.dumps(FAKE_SYNTHETIC)
    generate_synthetic_examples("test", fresh=True)
    _, kwargs = mock_ollama.call_args
    assert kwargs.get("temperature") == 0


# ── generate_explanations_batch ────────────────────────────

@patch("classifier.call_ollama")
def test_explanations_batch_parses(mock_ollama):
    mock_ollama.return_value = FAKE_EXPLANATIONS
    classifications = [True, False, False]
    result = generate_explanations_batch(SAMPLE_MSGS, classifications, "travel receipts")
    assert len(result) == 3
    assert "Delta" in result[0]
    assert mock_ollama.call_count == 1


@patch("classifier.call_ollama")
def test_explanations_batch_fallback(mock_ollama):
    mock_ollama.side_effect = Exception("connection failed")
    classifications = [True, False]
    result = generate_explanations_batch(SAMPLE_MSGS[:2], classifications, "travel")
    assert len(result) == 2
    assert "matches" in result[0]
    assert "does not match" in result[1]


@patch("classifier.call_ollama")
def test_explanations_batch_truncates_long(mock_ollama):
    long_explanation = "x" * 200
    mock_ollama.return_value = json.dumps([long_explanation])
    result = generate_explanations_batch(SAMPLE_MSGS[:1], [True], "test")
    assert len(result[0]) <= 153  # 150 + "..."


# ── classify (integration with mocked ollama) ──────────────

@patch("classifier.call_ollama")
def test_classify_returns_correct_schema(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),  # synthetic examples
        FAKE_EXPLANATIONS,           # batch explanations
    ]
    results = classify(SAMPLE_MSGS, "travel receipts", fresh=True)
    assert len(results) == 3
    for r in results:
        assert "message_id" in r
        assert "is_in_category" in r
        assert "explanation" in r
        assert isinstance(r["is_in_category"], bool)
        assert isinstance(r["explanation"], str)


@patch("classifier.call_ollama")
def test_classify_applies_feedback(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_EXPLANATIONS,
    ]
    feedback = {"174a9": {"corrected_label": False}}
    results = classify(SAMPLE_MSGS, "travel receipts", feedback=feedback, fresh=True)
    delta_result = next(r for r in results if r["message_id"] == "174a9")
    assert delta_result["is_in_category"] is False
    assert "(corrected)" in delta_result["explanation"]


@patch("classifier.call_ollama")
def test_classify_empty_messages(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        "[]",
    ]
    results = classify([], "travel receipts", fresh=True)
    assert results == []


@patch("classifier.call_ollama")
def test_classify_no_examples_returns_empty(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = "garbage"
    results = classify(SAMPLE_MSGS, "bad", fresh=True)
    assert results == []


# ── classify_multi (integration with mocked ollama) ────────

@patch("classifier.call_ollama")
def test_classify_multi_returns_all_categories(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),  # synthetic for cat 1
        json.dumps(FAKE_SYNTHETIC),  # synthetic for cat 2
        FAKE_EXPLANATIONS,           # explanations for cat 1
        FAKE_EXPLANATIONS,           # explanations for cat 2
    ]
    results = classify_multi(SAMPLE_MSGS, ["travel", "family"], fresh=True)
    assert "travel" in results
    assert "family" in results
    assert len(results["travel"]) == 3
    assert len(results["family"]) == 3


@patch("classifier.call_ollama")
def test_classify_multi_applies_per_category_feedback(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        json.dumps(FAKE_SYNTHETIC),
        FAKE_EXPLANATIONS,
        FAKE_EXPLANATIONS,
    ]
    feedbacks = {
        "travel": {"174a9": {"corrected_label": False}},
    }
    results = classify_multi(SAMPLE_MSGS, ["travel", "family"], feedbacks=feedbacks, fresh=True)
    delta_travel = next(r for r in results["travel"] if r["message_id"] == "174a9")
    delta_family = next(r for r in results["family"] if r["message_id"] == "174a9")
    assert delta_travel["is_in_category"] is False
    assert "(corrected)" not in delta_family["explanation"]


@patch("classifier.call_ollama")
def test_classify_multi_skips_bad_categories(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        "garbage",                   # bad synthetic for cat 1
        json.dumps(FAKE_SYNTHETIC),  # good synthetic for cat 2
        FAKE_EXPLANATIONS,           # explanations for cat 2
    ]
    results = classify_multi(SAMPLE_MSGS, ["bad", "good"], fresh=True)
    assert "bad" not in results
    assert "good" in results


# ── output schema matches spec ─────────────────────────────

@patch("classifier.call_ollama")
def test_output_matches_spec_schema(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_EXPLANATIONS,
    ]
    results = classify(SAMPLE_MSGS, "travel receipts", fresh=True)
    for r in results:
        assert set(r.keys()) == {"message_id", "is_in_category", "explanation"}
        assert isinstance(r["message_id"], str)
        assert isinstance(r["is_in_category"], bool)
        assert isinstance(r["explanation"], str)
        assert len(r["explanation"]) > 0


# ── margin threshold logic ─────────────────────────────────

def test_margin_threshold_defaults_to_no():
    yes_centroid = np.array([1.0, 0.0])
    no_centroid = np.array([0.98, 0.02])
    email_emb = np.array([0.99, 0.01])

    yes_sim = cosine_sim(email_emb, yes_centroid)
    no_sim = cosine_sim(email_emb, no_centroid)
    margin = abs(yes_sim - no_sim)

    if margin < MARGIN_THRESHOLD:
        is_match = False
    else:
        is_match = yes_sim > no_sim

    assert margin < MARGIN_THRESHOLD
    assert is_match is False


def test_clear_match_above_threshold():
    yes_centroid = np.array([1.0, 0.0])
    no_centroid = np.array([0.0, 1.0])
    email_emb = np.array([0.9, 0.1])

    yes_sim = cosine_sim(email_emb, yes_centroid)
    no_sim = cosine_sim(email_emb, no_centroid)
    margin = abs(yes_sim - no_sim)

    is_match = bool(yes_sim > no_sim) if margin >= MARGIN_THRESHOLD else False
    assert margin > MARGIN_THRESHOLD
    assert is_match is True
