#!/usr/bin/env python3
"""
gmail custom category builder

approach: synthetic contrastive examples
- ask local LLM (ollama) to generate synthetic "yes" and "no" example emails
- embed everything with sentence-transformers (free, local)
- classify each real email by cosine similarity to yes/no centroids

tried KNN and pipeline decomposition but centroid is more stable
with noisy synthetic data (see dev_journal.md)
"""

import json
import base64
import sys
import os
import re
import logging
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
MARGIN_THRESHOLD = 0.05


class OllamaConnectionError(Exception):
    pass


def load_messages(path):
    messages = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def decode_body(body_b64):
    try:
        raw = base64.b64decode(body_b64).decode("utf-8")
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    except Exception as e:
        return f"[decode error: {e}]"


def email_to_text(msg):
    body = decode_body(msg.get("body", ""))
    to = ", ".join(msg["to"]) if isinstance(msg.get("to"), list) else msg.get("to", "")
    parts = [
        f"From: {msg.get('from', '')}",
        f"To: {to}",
        f"Date: {msg.get('date', '')}",
        f"Subject: {msg.get('subject', '')}",
        f"Snippet: {msg.get('snippet', '')}",
        f"Body: {body}",
    ]
    return "\n".join(parts)


def call_ollama(prompt, temperature=0.7):
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 8000},
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.ConnectionError:
        raise OllamaConnectionError(
            f"cant connect to ollama at {OLLAMA_URL}. "
            f"make sure ollama is running (ollama serve) and {OLLAMA_MODEL} is pulled"
        )
    except requests.HTTPError as e:
        raise OllamaConnectionError(f"ollama returned an error: {e}")
    except (KeyError, requests.JSONDecodeError):
        raise OllamaConnectionError(
            f"unexpected response from ollama — is {OLLAMA_MODEL} pulled?"
        )


def parse_json_response(raw_text):
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```\w*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'[\[{].*[\]}]', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    obj_matches = re.findall(r'\{[^{}]+\}', raw_text)
    recovered = []
    for obj_str in obj_matches:
        try:
            recovered.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue
    if recovered:
        logger.info("recovered %d objects from partial json", len(recovered))
        return recovered

    logger.warning("failed to parse json: %s", raw_text[:300])
    return None


def _parse_examples(examples):
    yes_texts = []
    no_texts = []
    edge_texts = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        text = f"From: {ex.get('from', '')}\nSubject: {ex.get('subject', '')}\nSnippet: {ex.get('snippet', '')}\nBody: {ex.get('body', '')}"
        label = str(ex.get("label", "")).lower().strip()
        if label == "yes":
            yes_texts.append(text)
        elif label == "no":
            no_texts.append(text)
        elif label == "edge":
            edge_texts.append(text)
    return yes_texts, no_texts, edge_texts


def generate_synthetic_examples(category_description, fresh=False):
    """generate synthetic yes/no emails for this category. caches to disk."""
    safe_name = re.sub(r"[^a-z0-9]+", "_", category_description.lower()).strip("_")
    cache_path = f"synthetic_{safe_name}.json"

    if not fresh and os.path.exists(cache_path):
        logger.info("loading cached examples from %s", cache_path)
        with open(cache_path, "r") as f:
            examples = json.load(f)
        yes_texts, no_texts, edge_texts = _parse_examples(examples)
        if yes_texts and no_texts:
            logger.info("got %d yes, %d no, %d edge examples (cached)", len(yes_texts), len(no_texts), len(edge_texts))
            return yes_texts, no_texts, edge_texts
        logger.info("cache was bad, regenerating...")

    prompt = f"""Generate synthetic emails for this category: "{category_description}"

JSON array, 12 objects. Keys: label, from, subject, body, reason. Keep body under 10 words.
- 5 label "yes" (clearly match — each from a DIFFERENT sender with DIFFERENT subject. no duplicates)
- 5 label "no" (clearly dont match — make VERY DIVERSE: one shopping, one social, one work/calendar, one newsletter, one notification. all different types)
- 2 label "edge" (tricky near-misses that look similar but shouldnt match)

IMPORTANT: every single example must have a unique sender and subject. no repeats.
Output ONLY the JSON array:"""

    logger.info("generating synthetic examples...")
    raw = call_ollama(prompt, temperature=0)
    examples = parse_json_response(raw)

    if not examples or not isinstance(examples, list):
        return [], [], []

    yes_texts, no_texts, edge_texts = _parse_examples(examples)
    logger.info("got %d yes, %d no, %d edge examples", len(yes_texts), len(no_texts), len(edge_texts))

    with open(cache_path, "w") as f:
        json.dump(examples, f, indent=2)

    return yes_texts, no_texts, edge_texts


def generate_heuristic_rules(category_description, fresh=False):
    """ask ollama for simple keyword/sender rules for this category. caches to disk."""
    safe_name = re.sub(r"[^a-z0-9]+", "_", category_description.lower()).strip("_")
    cache_path = f"rules_{safe_name}.json"

    if not fresh and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            rules = json.load(f)
        if rules.get("keywords"):
            logger.info("loaded %d keyword rules (cached)", len(rules["keywords"]))
            return rules

    prompt = f"""Category: "{category_description}"

Generate simple heuristic rules for detecting emails in this category.
Output a JSON object with these keys:
- "keywords": list of 5-8 words/phrases that would appear in matching emails (subject or body)
- "sender_hints": list of 3-5 sender names or domains that would send these emails
- "exclude_keywords": list of 3-5 words that indicate an email does NOT match even if it looks similar

Be specific to this category. Output ONLY the JSON object:"""

    logger.info("generating heuristic rules...")
    raw = call_ollama(prompt, temperature=0)
    rules = parse_json_response(raw)

    if not isinstance(rules, dict) or "keywords" not in rules:
        rules = {"keywords": [], "sender_hints": [], "exclude_keywords": []}

    for key in ("keywords", "sender_hints", "exclude_keywords"):
        if key not in rules or not isinstance(rules[key], list):
            rules[key] = []
        rules[key] = [str(item).lower().strip() for item in rules[key] if item]

    exclude_set = set(rules["exclude_keywords"])
    rules["keywords"] = [k for k in rules["keywords"] if k not in exclude_set]

    with open(cache_path, "w") as f:
        json.dump(rules, f, indent=2)

    return rules


def score_rules(msg, rules):
    """score an email against heuristic rules. returns a float."""
    text = f"{msg.get('subject', '')} {msg.get('snippet', '')}".lower()
    body = decode_body(msg.get("body", "")).lower()
    sender = msg.get("from", "").lower()
    full_text = f"{text} {body}"

    score = 0
    for kw in rules.get("keywords", []):
        if kw in full_text:
            score += 1
    for hint in rules.get("sender_hints", []):
        if hint in sender:
            score += 1
    for excl in rules.get("exclude_keywords", []):
        if excl in full_text:
            score -= 1

    return score


def cosine_sim(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def generate_explanations_batch(messages, classifications, category_description):
    """generate explanations for all emails in a single LLM call, matched by message ID"""
    if not messages:
        return []

    lines = []
    msg_ids = []
    for msg, is_match in zip(messages, classifications):
        tag = "YES" if is_match else "NO"
        msg_id = msg.get("id", "")
        msg_ids.append(msg_id)
        lines.append(f'{msg_id} [{tag}] From: {msg.get("from", "")} Subject: "{msg.get("subject", "")}"')

    prompt = f"""Category: "{category_description}"

For each email below, write ONE short sentence (under 15 words) explaining why it matches or doesn't match the category. Mention the sender or subject.

{chr(10).join(lines)}

Output a JSON object mapping each email ID to its explanation string. Output ONLY the JSON object:"""

    try:
        raw = call_ollama(prompt)
        parsed = parse_json_response(raw)

        if isinstance(parsed, dict):
            explanations = []
            for i, msg_id in enumerate(msg_ids):
                exp = str(parsed.get(msg_id, parsed.get(str(msg_id), ""))).strip().strip('"').strip("'")
                if not exp:
                    is_match = classifications[i]
                    exp = f"{'matches' if is_match else 'does not match'} \"{category_description}\""
                if len(exp) > 150:
                    exp = exp[:150] + "..."
                explanations.append(exp)
            return explanations

        if isinstance(parsed, list) and len(parsed) >= len(messages):
            explanations = []
            for item in parsed[:len(messages)]:
                exp = str(item).strip().strip('"').strip("'")
                if len(exp) > 150:
                    exp = exp[:150] + "..."
                explanations.append(exp)
            return explanations
    except Exception:
        pass

    return [
        f"{'matches' if is_match else 'does not match'} \"{category_description}\""
        for is_match in classifications
    ]


def get_feedback_path(category_description):
    safe_name = re.sub(r"[^a-z0-9]+", "_", category_description.lower()).strip("_")
    return f"feedback_{safe_name}.json"


def load_feedback(category_description):
    path = get_feedback_path(category_description)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_feedback(category_description, feedback):
    path = get_feedback_path(category_description)
    with open(path, "w") as f:
        json.dump(feedback, f, indent=2)


def classify(messages, category_description, feedback=None, fresh=False):
    logger.info("classifier: category=%s", category_description)

    yes_texts, no_texts, edge_texts = generate_synthetic_examples(category_description, fresh=fresh)
    if not yes_texts or not no_texts:
        logger.error("couldnt generate examples")
        return []

    rules = generate_heuristic_rules(category_description, fresh=fresh)

    logger.info("loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("embedding synthetic examples...")
    yes_emb = model.encode(yes_texts)
    no_all = no_texts + edge_texts
    no_emb = model.encode(no_all)

    yes_centroid = np.mean(yes_emb, axis=0)
    no_centroid = np.mean(no_emb, axis=0)

    email_texts = [email_to_text(msg) for msg in messages]
    logger.info("batch embedding %d emails...", len(messages))
    all_emb = model.encode(email_texts)

    logger.info("classifying...")
    classifications = []
    rule_tiebreaks = 0

    for msg, email_emb in zip(messages, all_emb):
        yes_sim = cosine_sim(email_emb, yes_centroid)
        no_sim = cosine_sim(email_emb, no_centroid)
        margin = abs(yes_sim - no_sim)

        if margin < MARGIN_THRESHOLD:
            rscore = score_rules(msg, rules)
            if rscore > 0:
                is_match = True
                rule_tiebreaks += 1
            else:
                is_match = False
        else:
            is_match = bool(yes_sim > no_sim)

        classifications.append(is_match)
        status = "YES" if is_match else " no"
        logger.info("[%s] %s: %s", status, msg["id"], msg["subject"][:50])

    if rule_tiebreaks:
        logger.info("rules broke %d tiebreak(s)", rule_tiebreaks)

    logger.info("generating explanations (batch)...")
    explanations = generate_explanations_batch(messages, classifications, category_description)

    results = []
    for msg, is_match, explanation in zip(messages, classifications, explanations):
        result = {
            "message_id": msg["id"],
            "is_in_category": bool(is_match),
            "explanation": explanation,
        }
        results.append(result)
        logger.info("  %s: %s", msg["id"], explanation)

    if feedback:
        applied = 0
        for result in results:
            if result["message_id"] in feedback:
                corrected = feedback[result["message_id"]]["corrected_label"]
                if result["is_in_category"] != corrected:
                    result["is_in_category"] = corrected
                    result["explanation"] += " (corrected)"
                    applied += 1
        if applied:
            logger.info("applied %d corrections", applied)

    return results


def classify_multi(messages, categories, feedbacks=None, fresh=False):
    """classify emails against multiple categories, reusing model + embeddings"""
    if feedbacks is None:
        feedbacks = {}

    logger.info("multi-category classifier: %s", categories)

    category_examples = {}
    category_rules = {}
    for cat in categories:
        yes_texts, no_texts, edge_texts = generate_synthetic_examples(cat, fresh=fresh)
        if not yes_texts or not no_texts:
            logger.warning("couldnt generate examples for '%s', skipping", cat)
            continue
        category_examples[cat] = (yes_texts, no_texts, edge_texts)
        category_rules[cat] = generate_heuristic_rules(cat, fresh=fresh)

    if not category_examples:
        logger.error("no valid categories")
        return {}

    logger.info("loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    email_texts = [email_to_text(msg) for msg in messages]
    logger.info("batch embedding %d emails...", len(messages))
    all_emb = model.encode(email_texts)

    all_results = {}
    for cat, (yes_texts, no_texts, edge_texts) in category_examples.items():
        logger.info("--- %s ---", cat)
        rules = category_rules[cat]

        yes_emb = model.encode(yes_texts)
        no_all = no_texts + edge_texts
        no_emb = model.encode(no_all)

        yes_centroid = np.mean(yes_emb, axis=0)
        no_centroid = np.mean(no_emb, axis=0)

        classifications = []
        rule_tiebreaks = 0
        for msg, email_emb in zip(messages, all_emb):
            yes_sim = cosine_sim(email_emb, yes_centroid)
            no_sim = cosine_sim(email_emb, no_centroid)
            margin = abs(yes_sim - no_sim)

            if margin < MARGIN_THRESHOLD:
                rscore = score_rules(msg, rules)
                if rscore > 0:
                    is_match = True
                    rule_tiebreaks += 1
                else:
                    is_match = False
            else:
                is_match = bool(yes_sim > no_sim)

            classifications.append(is_match)
            status = "YES" if is_match else " no"
            logger.info("[%s] %s: %s", status, msg["id"], msg["subject"][:50])

        if rule_tiebreaks:
            logger.info("rules broke %d tiebreak(s)", rule_tiebreaks)

        logger.info("generating explanations (batch)...")
        explanations = generate_explanations_batch(messages, classifications, cat)

        results = []
        for msg, is_match, explanation in zip(messages, classifications, explanations):
            result = {
                "message_id": msg["id"],
                "is_in_category": bool(is_match),
                "explanation": explanation,
            }
            results.append(result)
            logger.info("  %s: %s", msg["id"], explanation)

        feedback = feedbacks.get(cat, {})
        if feedback:
            applied = 0
            for result in results:
                if result["message_id"] in feedback:
                    corrected = feedback[result["message_id"]]["corrected_label"]
                    if result["is_in_category"] != corrected:
                        result["is_in_category"] = corrected
                        result["explanation"] += " (corrected)"
                        applied += 1
            if applied:
                logger.info("applied %d corrections for '%s'", applied, cat)

        all_results[cat] = results

    return all_results


def run(messages_path, category_description, feedback=None):
    messages = load_messages(messages_path)
    results = classify(messages, category_description, feedback)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    messages_file = sys.argv[1] if len(sys.argv) > 1 else "messages.jsonl"
    try:
        if len(sys.argv) > 3:
            messages = load_messages(messages_file)
            all_results = classify_multi(messages, sys.argv[2:])
            print(json.dumps(all_results, indent=2))
        elif len(sys.argv) == 3:
            run(messages_file, sys.argv[2])
        else:
            category = input("enter category description: ")
            run(messages_file, category)
    except OllamaConnectionError as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        sys.exit(1)
