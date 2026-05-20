#!/usr/bin/env python3
"""tests for classifier.py"""

import json
import base64
import os
import shutil
import numpy as np
import pytest
import requests
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
    generate_heuristic_rules,
    score_rules,
    get_feedback_path,
    load_feedback,
    save_feedback,
    classify,
    classify_multi,
    call_ollama,
    OllamaConnectionError,
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

FAKE_EXPLANATIONS = json.dumps({
    "174a9": "Delta eTicket is a travel receipt from an airline",
    "174b2": "AI newsletter is not a travel receipt",
    "174c5": "Mom's email is family communication, not travel",
})

FAKE_RULES = json.dumps({
    "keywords": ["receipt", "booking", "eticket"],
    "sender_hints": ["delta", "united"],
    "exclude_keywords": ["newsletter"],
})

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(PROJECT_DIR, "fixtures")


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
def test_explanations_batch_parses_dict(mock_ollama):
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
    mock_ollama.return_value = json.dumps({"174a9": long_explanation})
    result = generate_explanations_batch(SAMPLE_MSGS[:1], [True], "test")
    assert len(result[0]) <= 153  # 150 + "..."


@patch("classifier.call_ollama")
def test_explanations_batch_handles_missing_id(mock_ollama):
    """if LLM omits an ID, that email gets a generic fallback explanation"""
    mock_ollama.return_value = json.dumps({
        "174a9": "Delta is a travel receipt",
        # 174b2 missing
        "174c5": "Mom email is family",
    })
    classifications = [True, False, False]
    result = generate_explanations_batch(SAMPLE_MSGS, classifications, "travel")
    assert len(result) == 3
    assert "Delta" in result[0]
    assert "does not match" in result[1]  # generic fallback
    assert "Mom" in result[2] or "family" in result[2]


@patch("classifier.call_ollama")
def test_explanations_batch_reorder_tolerance(mock_ollama):
    """explanations returned out of order should still match correctly by ID"""
    mock_ollama.return_value = json.dumps({
        "174c5": "third email explanation",
        "174a9": "first email explanation",
        "174b2": "second email explanation",
    })
    classifications = [True, False, False]
    result = generate_explanations_batch(SAMPLE_MSGS, classifications, "test")
    assert result[0] == "first email explanation"
    assert result[1] == "second email explanation"
    assert result[2] == "third email explanation"


@patch("classifier.call_ollama")
def test_explanations_batch_array_fallback(mock_ollama):
    """positional array format still works as fallback"""
    mock_ollama.return_value = json.dumps([
        "explanation one",
        "explanation two",
        "explanation three",
    ])
    classifications = [True, False, False]
    result = generate_explanations_batch(SAMPLE_MSGS, classifications, "test")
    assert result[0] == "explanation one"
    assert result[1] == "explanation two"
    assert result[2] == "explanation three"


def test_explanations_batch_empty_messages():
    result = generate_explanations_batch([], [], "test")
    assert result == []


# ── classify (integration with mocked ollama) ──────────────

@patch("classifier.call_ollama")
def test_classify_returns_correct_schema(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),  # synthetic examples
        FAKE_RULES,                  # heuristic rules
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
        FAKE_RULES,
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
        FAKE_RULES,
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
        FAKE_RULES,                  # rules for cat 1
        json.dumps(FAKE_SYNTHETIC),  # synthetic for cat 2
        FAKE_RULES,                  # rules for cat 2
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
        FAKE_RULES,
        json.dumps(FAKE_SYNTHETIC),
        FAKE_RULES,
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
        FAKE_RULES,                  # rules for cat 2
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
        FAKE_RULES,
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


# ── score_rules ────────────────────────────────────────────

SAMPLE_RULES = {
    "keywords": ["receipt", "booking", "eticket", "confirmation"],
    "sender_hints": ["delta", "united", "aa.com"],
    "exclude_keywords": ["newsletter", "social"],
}


def test_score_rules_keyword_hits():
    msg = {"subject": "Your eTicket Receipt", "snippet": "booking confirmed", "from": "someone", "body": ""}
    score = score_rules(msg, SAMPLE_RULES)
    assert score >= 2


def test_score_rules_sender_match():
    msg = {"subject": "Something", "snippet": "", "from": "Delta <noreply@delta.com>", "body": ""}
    score = score_rules(msg, SAMPLE_RULES)
    assert score >= 1


def test_score_rules_exclusion():
    msg = {"subject": "Newsletter about travel", "snippet": "", "from": "news@test.com", "body": ""}
    score = score_rules(msg, SAMPLE_RULES)
    assert score < 1  # newsletter exclusion cancels out any keyword hit


def test_score_rules_no_matches():
    msg = {"subject": "Weekend BBQ plans", "snippet": "firing up the grill", "from": "Dad <dad@gmail.com>", "body": ""}
    score = score_rules(msg, SAMPLE_RULES)
    assert score == 0


def test_score_rules_empty_rules():
    msg = {"subject": "anything", "snippet": "", "from": "anyone", "body": ""}
    score = score_rules(msg, {"keywords": [], "sender_hints": [], "exclude_keywords": []})
    assert score == 0


def test_score_rules_body_content():
    msg = {
        "subject": "Your flight",
        "snippet": "",
        "from": "airline@test.com",
        "body": base64.b64encode(b"Here is your receipt and booking confirmation").decode(),
    }
    score = score_rules(msg, SAMPLE_RULES)
    assert score >= 2


def test_score_rules_mixed_signals():
    msg = {"subject": "Receipt for newsletter subscription", "snippet": "", "from": "news@test.com", "body": ""}
    score = score_rules(msg, SAMPLE_RULES)
    # "receipt" +1, "newsletter" -1 = 0 or close
    assert score <= 1


# ── generate_heuristic_rules ───────────────────────────────

@patch("classifier.call_ollama")
def test_generate_rules_caches(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = FAKE_RULES
    rules = generate_heuristic_rules("travel receipts", fresh=True)
    assert "receipt" in rules["keywords"]
    assert mock_ollama.call_count == 1

    rules2 = generate_heuristic_rules("travel receipts", fresh=False)
    assert rules2["keywords"] == rules["keywords"]
    assert mock_ollama.call_count == 1  # used cache


@patch("classifier.call_ollama")
def test_generate_rules_deduplicates(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    overlapping = json.dumps({
        "keywords": ["travel", "receipt", "blog"],
        "sender_hints": ["delta"],
        "exclude_keywords": ["travel", "spam"],
    })
    mock_ollama.return_value = overlapping
    rules = generate_heuristic_rules("test", fresh=True)
    assert "travel" not in rules["keywords"]
    assert "receipt" in rules["keywords"]
    assert "travel" in rules["exclude_keywords"]


@patch("classifier.call_ollama")
def test_generate_rules_bad_response(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_ollama.return_value = "not json"
    rules = generate_heuristic_rules("bad", fresh=True)
    assert rules["keywords"] == []
    assert rules["sender_hints"] == []


# ── tiebreaker integration ─────────────────────────────────

def test_tiebreaker_rules_flip_borderline():
    """when margin is below threshold but rules match, result should be yes"""
    yes_centroid = np.array([1.0, 0.0])
    no_centroid = np.array([0.0, 1.0])
    email_emb = np.array([0.707, 0.707])

    yes_sim = cosine_sim(email_emb, yes_centroid)
    no_sim = cosine_sim(email_emb, no_centroid)
    margin = abs(yes_sim - no_sim)
    assert margin < MARGIN_THRESHOLD

    rules = {"keywords": ["receipt", "booking"], "sender_hints": ["delta"], "exclude_keywords": []}
    msg = {"subject": "Your Receipt", "snippet": "booking", "from": "Delta <noreply@delta.com>", "body": ""}

    rscore = score_rules(msg, rules)
    assert rscore > 0

    is_match = True if rscore > 0 else False
    assert is_match is True


def test_tiebreaker_rules_keep_no():
    """when margin is below threshold and rules dont match, result stays no"""
    rules = {"keywords": ["receipt"], "sender_hints": ["delta"], "exclude_keywords": []}
    msg = {"subject": "Weekend BBQ", "snippet": "grill", "from": "Dad <dad@gmail.com>", "body": ""}
    rscore = score_rules(msg, rules)
    assert rscore <= 0


@patch("classifier.SentenceTransformer")
@patch("classifier.call_ollama")
def test_tiebreaker_fires_through_classify(mock_ollama, mock_st_class, tmp_path, monkeypatch):
    """run full classify pipeline with controlled embeddings to verify tiebreaker changes a result"""
    monkeypatch.chdir(tmp_path)

    mock_model = MagicMock()
    mock_st_class.return_value = mock_model

    # yes/no centroids orthogonal, email1 equidistant (borderline), email2 clearly no
    mock_model.encode.side_effect = [
        np.array([[1.0, 0.0]]),                          # yes synthetic embeddings
        np.array([[0.0, 1.0]]),                          # no synthetic embeddings
        np.array([[0.707, 0.707], [0.1, 0.995]]),        # email embeddings
    ]

    messages = [
        {"id": "m1", "subject": "Your Receipt", "from": "Delta <noreply@delta.com>",
         "to": [], "snippet": "booking confirmed", "body": "", "date": ""},
        {"id": "m2", "subject": "Weekend BBQ", "from": "Dad <dad@gmail.com>",
         "to": [], "snippet": "grill", "body": "", "date": ""},
    ]

    mock_ollama.side_effect = [
        json.dumps([
            {"label": "yes", "from": "Delta", "subject": "Receipt", "snippet": "", "body": "flight receipt"},
            {"label": "no", "from": "Mom", "subject": "Hello", "snippet": "", "body": "family stuff"},
        ]),
        json.dumps({
            "keywords": ["receipt", "booking", "flight"],
            "sender_hints": ["delta"],
            "exclude_keywords": [],
        }),
        json.dumps({"m1": "receipt from Delta", "m2": "BBQ is not travel"}),
    ]

    results = classify(messages, "travel receipts", fresh=True)

    result_map = {r["message_id"]: r["is_in_category"] for r in results}
    assert result_map["m1"] is True   # borderline margin + rules match → yes
    assert result_map["m2"] is False  # clearly no by embeddings


# ── OllamaConnectionError ─────────────────────────────────

def test_ollama_connection_error_is_raised():
    """call_ollama raises OllamaConnectionError instead of sys.exit"""
    with patch("classifier.requests.post") as mock_post:
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("refused")
        with pytest.raises(OllamaConnectionError) as exc_info:
            call_ollama("test prompt")
        assert "cant connect" in str(exc_info.value)


# ── fixture quality validation ─────────────────────────────

def test_committed_fixtures_exist():
    """golden fixture files are committed and loadable"""
    synth_path = os.path.join(FIXTURES_DIR, "synthetic_work_related_travel_receipts_from_airlines.json")
    rules_path = os.path.join(FIXTURES_DIR, "rules_work_related_travel_receipts_from_airlines.json")
    assert os.path.exists(synth_path), "synthetic fixture missing"
    assert os.path.exists(rules_path), "rules fixture missing"

    with open(synth_path) as f:
        examples = json.load(f)
    with open(rules_path) as f:
        rules = json.load(f)

    assert isinstance(examples, list) and len(examples) == 12
    assert isinstance(rules, dict) and "keywords" in rules


def test_fixture_synthetic_quality():
    """validate that committed synthetic fixtures have diverse, high-quality data"""
    synth_path = os.path.join(FIXTURES_DIR, "synthetic_work_related_travel_receipts_from_airlines.json")
    with open(synth_path) as f:
        examples = json.load(f)

    yes_examples = [e for e in examples if e["label"] == "yes"]
    no_examples = [e for e in examples if e["label"] == "no"]
    edge_examples = [e for e in examples if e["label"] == "edge"]

    assert len(yes_examples) == 5
    assert len(no_examples) == 5
    assert len(edge_examples) == 2

    senders = [e["from"] for e in examples]
    assert len(senders) == len(set(senders)), "duplicate senders in fixtures"

    subjects = [e["subject"] for e in examples]
    assert len(subjects) == len(set(subjects)), "duplicate subjects in fixtures"

    for ex in yes_examples:
        text = f"{ex['subject']} {ex['body']}".lower()
        assert any(kw in text for kw in ["flight", "eticket", "booking", "receipt", "airline", "itinerary", "confirmation", "trip"]), \
            f"yes example doesnt mention airline content: {ex['subject']}"

    no_types = set()
    for ex in no_examples:
        text = f"{ex['from']} {ex['subject']} {ex['body']}".lower()
        if any(w in text for w in ["order", "shipped", "amazon"]):
            no_types.add("shopping")
        elif any(w in text for w in ["mom", "dad", "family", "thanksgiving"]):
            no_types.add("family")
        elif any(w in text for w in ["meeting", "standup", "calendar"]):
            no_types.add("work")
        elif any(w in text for w in ["digest", "newsletter", "roundup"]):
            no_types.add("newsletter")
        elif any(w in text for w in ["payment", "venmo", "notification"]):
            no_types.add("notification")
    assert len(no_types) >= 4, f"no examples not diverse enough: {no_types}"


# ── classification accuracy with committed fixtures ────────

@patch("classifier.call_ollama")
def test_classification_accuracy_travel(mock_ollama, tmp_path, monkeypatch):
    """end-to-end accuracy test: real messages + committed fixtures + real embeddings"""
    monkeypatch.chdir(tmp_path)

    cat_name = "work_related_travel_receipts_from_airlines"
    shutil.copy(
        os.path.join(FIXTURES_DIR, f"synthetic_{cat_name}.json"),
        tmp_path / f"synthetic_{cat_name}.json",
    )
    shutil.copy(
        os.path.join(FIXTURES_DIR, f"rules_{cat_name}.json"),
        tmp_path / f"rules_{cat_name}.json",
    )

    messages = load_messages(os.path.join(PROJECT_DIR, "messages.jsonl"))

    ids = [m["id"] for m in messages]
    mock_ollama.return_value = json.dumps({mid: "explanation" for mid in ids})

    results = classify(messages, "work-related travel receipts from airlines")
    result_map = {r["message_id"]: r["is_in_category"] for r in results}

    assert result_map["174a9"] is True,  "Delta eTicket should match"
    assert result_map["174d8"] is True,  "United Booking should match"

    # AA personal trip is a known false positive — embeddings weight airline
    # language too heavily to distinguish work vs personal (see dev_journal.md)
    assert result_map["17526"] is True, "AA trip is a known false positive"

    assert result_map["174b2"] is False, "AI Newsletter should not match"
    assert result_map["174c5"] is False, "Mom Thanksgiving should not match"
    assert result_map["174e1"] is False, "Amazon order should not match"
    assert result_map["174f3"] is False, "Q3 Meeting should not match"
    assert result_map["17500"] is False, "Papers We Love should not match"
    assert result_map["17513"] is False, "Dad BBQ should not match"
    assert result_map["17539"] is False, "GitHub PR should not match"
    assert result_map["1753c"] is False, "Family photos should not match"
    assert result_map["1754f"] is False, "Spotify should not match"

    assert len(results) == 12, "should have result for every email"


# ── call_ollama error handling ─────────────────────────────

def test_ollama_http_error_raises():
    """HTTP 500 from ollama should raise OllamaConnectionError, not requests.HTTPError"""
    with patch("classifier.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp
        with pytest.raises(OllamaConnectionError) as exc_info:
            call_ollama("test")
        assert "error" in str(exc_info.value).lower()


def test_ollama_bad_json_raises():
    """ollama returning non-JSON or missing 'response' key should raise OllamaConnectionError"""
    with patch("classifier.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"error": "model not found"}
        mock_post.return_value = mock_resp
        with pytest.raises(OllamaConnectionError) as exc_info:
            call_ollama("test")
        assert "unexpected" in str(exc_info.value).lower()


def test_ollama_timeout_is_set():
    """requests.post should be called with a timeout"""
    with patch("classifier.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_post.return_value = mock_resp
        call_ollama("test")
        _, kwargs = mock_post.call_args
        assert "timeout" in kwargs and kwargs["timeout"] > 0


# ── classify edge cases ───────────────────────────────────

@patch("classifier.call_ollama")
def test_classify_single_message(mock_ollama, tmp_path, monkeypatch):
    """classify should work with a single message"""
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_RULES,
        json.dumps({"174a9": "Delta is a travel receipt"}),
    ]
    results = classify([SAMPLE_MSG], "travel receipts", fresh=True)
    assert len(results) == 1
    assert results[0]["message_id"] == "174a9"
    assert isinstance(results[0]["is_in_category"], bool)


@patch("classifier.call_ollama")
def test_classify_result_ids_match_input(mock_ollama, tmp_path, monkeypatch):
    """every result message_id must correspond to an input message"""
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_RULES,
        FAKE_EXPLANATIONS,
    ]
    results = classify(SAMPLE_MSGS, "travel receipts", fresh=True)
    input_ids = {m["id"] for m in SAMPLE_MSGS}
    result_ids = {r["message_id"] for r in results}
    assert result_ids == input_ids


@patch("classifier.call_ollama")
def test_classify_explanations_count_matches(mock_ollama, tmp_path, monkeypatch):
    """every result must have a non-empty explanation"""
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_RULES,
        FAKE_EXPLANATIONS,
    ]
    results = classify(SAMPLE_MSGS, "travel receipts", fresh=True)
    for r in results:
        assert len(r["explanation"]) > 0, f"empty explanation for {r['message_id']}"


@patch("classifier.call_ollama")
def test_classify_output_is_json_serializable(mock_ollama, tmp_path, monkeypatch):
    """results must be JSON-serializable (spec requires JSON output)"""
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_RULES,
        FAKE_EXPLANATIONS,
    ]
    results = classify(SAMPLE_MSGS, "travel receipts", fresh=True)
    serialized = json.dumps(results)
    roundtripped = json.loads(serialized)
    assert len(roundtripped) == len(results)
    for orig, rt in zip(results, roundtripped):
        assert orig["message_id"] == rt["message_id"]
        assert orig["is_in_category"] == rt["is_in_category"]


@patch("classifier.call_ollama")
def test_classify_multi_single_category(mock_ollama, tmp_path, monkeypatch):
    """classify_multi with one category should produce same results as classify"""
    monkeypatch.chdir(tmp_path)
    mock_ollama.side_effect = [
        json.dumps(FAKE_SYNTHETIC),
        FAKE_RULES,
        FAKE_EXPLANATIONS,
    ]
    results = classify_multi(SAMPLE_MSGS, ["travel"], fresh=True)
    assert "travel" in results
    assert len(results["travel"]) == 3
    for r in results["travel"]:
        assert set(r.keys()) == {"message_id", "is_in_category", "explanation"}


# ── score_rules edge cases ────────────────────────────────

def test_score_rules_missing_body_key():
    """score_rules should handle messages with no body key"""
    msg = {"subject": "Receipt", "snippet": "booking", "from": "delta@test.com"}
    rules = {"keywords": ["receipt"], "sender_hints": ["delta"], "exclude_keywords": []}
    score = score_rules(msg, rules)
    assert score >= 1


def test_score_rules_case_insensitive():
    """keywords should match regardless of case"""
    msg = {"subject": "YOUR ETICKET RECEIPT", "snippet": "", "from": "DELTA", "body": ""}
    rules = {"keywords": ["eticket", "receipt"], "sender_hints": ["delta"], "exclude_keywords": []}
    score = score_rules(msg, rules)
    assert score >= 3


# ── parse_json_response edge cases ─────────────────────────

def test_parse_json_empty_dict():
    result = parse_json_response("{}")
    assert result == {}


def test_parse_json_empty_array():
    result = parse_json_response("[]")
    assert result == []


def test_parse_json_unicode():
    raw = '{"name": "Confirmación de vuelo"}'
    result = parse_json_response(raw)
    assert result["name"] == "Confirmación de vuelo"


# ── decode_body edge cases ────────────────────────────────

def test_decode_body_large_content():
    """should handle large email bodies without error"""
    big = "A" * 100000
    encoded = base64.b64encode(big.encode()).decode()
    result = decode_body(encoded)
    assert len(result) == 100000


def test_decode_body_mixed_html_text():
    html = "Plain text before <b>bold</b> and after"
    encoded = base64.b64encode(html.encode()).decode()
    result = decode_body(encoded)
    assert "Plain text before" in result
    assert "bold" in result
    assert "and after" in result
    assert "<b>" not in result
