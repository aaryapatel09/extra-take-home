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
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
MARGIN_THRESHOLD = 0.05


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
    parts = [
        f"From: {msg.get('from', '')}",
        f"Subject: {msg.get('subject', '')}",
        f"Snippet: {msg.get('snippet', '')}",
        f"Body: {body}",
    ]
    return "\n".join(parts)


def call_ollama(prompt):
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 8000},
    })
    resp.raise_for_status()
    return resp.json()["response"]


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
        print(f"  recovered {len(recovered)} objects from partial json")
        return recovered

    print(f"  FAILED to parse json. raw:")
    print(raw_text[:300])
    return None


def generate_synthetic_examples(category_description):
    """generate synthetic yes/no emails for this category"""
    prompt = f"""Generate synthetic emails for this category: "{category_description}"

JSON array, 12 objects. Keys: label, from, subject, body, reason. Keep body under 10 words.
- 5 label "yes" (clearly match — each from a DIFFERENT sender with DIFFERENT subject. no duplicates)
- 5 label "no" (clearly dont match — make VERY DIVERSE: one shopping, one social, one work/calendar, one newsletter, one notification. all different types)
- 2 label "edge" (tricky near-misses that look similar but shouldnt match)

IMPORTANT: every single example must have a unique sender and subject. no repeats.
Output ONLY the JSON array:"""

    print("  generating synthetic examples...")
    raw = call_ollama(prompt)
    examples = parse_json_response(raw)

    if not examples or not isinstance(examples, list):
        return [], [], []

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

    print(f"  got {len(yes_texts)} yes, {len(no_texts)} no, {len(edge_texts)} edge examples")

    safe_name = re.sub(r"[^a-z0-9]+", "_", category_description.lower()).strip("_")
    with open(f"synthetic_{safe_name}.json", "w") as f:
        json.dump(examples, f, indent=2)

    return yes_texts, no_texts, edge_texts


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_explanation(msg, category_description, is_match):
    """ask ollama for a short plain english explanation"""
    if is_match:
        prompt = f"""Category: "{category_description}"
Email from {msg.get('from', '')} with subject "{msg.get('subject', '')}"
This email MATCHES the category. Write one short sentence (under 15 words) explaining why it matches. Mention the sender or subject. Just the sentence, nothing else."""
    else:
        prompt = f"""Category: "{category_description}"
Email from {msg.get('from', '')} with subject "{msg.get('subject', '')}"
This email DOES NOT match the category. Write one short sentence (under 15 words) explaining why it doesn't match. Mention the sender or subject. Just the sentence, nothing else."""

    try:
        raw = call_ollama(prompt)
        explanation = raw.strip().split("\n")[0].strip('"').strip("'").strip()
        if len(explanation) > 150:
            explanation = explanation[:150] + "..."
        return explanation
    except Exception:
        return f"{'matches' if is_match else 'does not match'} category"


def classify(messages, category_description):
    print(f"\n=== classifier ===")
    print(f"category: \"{category_description}\"\n")

    yes_texts, no_texts, edge_texts = generate_synthetic_examples(category_description)
    if not yes_texts or not no_texts:
        print("ERROR: couldnt generate examples")
        return []

    print("  loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  embedding synthetic examples...")
    yes_emb = model.encode(yes_texts)
    no_emb = model.encode(no_texts)

    yes_centroid = np.mean(yes_emb, axis=0)
    no_centroid = np.mean(no_emb, axis=0)

    print(f"  classifying {len(messages)} emails...\n")
    results = []

    for msg in messages:
        email_text = email_to_text(msg)
        email_emb = model.encode([email_text])[0]

        yes_sim = cosine_sim(email_emb, yes_centroid)
        no_sim = cosine_sim(email_emb, no_centroid)

        margin = abs(yes_sim - no_sim)

        if margin < MARGIN_THRESHOLD:
            is_match = False
        else:
            is_match = yes_sim > no_sim

        status = "YES" if is_match else " no"
        print(f"  [{status}] {msg['id']}: {msg['subject'][:50]}")

        explanation = generate_explanation(msg, category_description, is_match)

        result = {
            "message_id": msg["id"],
            "is_in_category": bool(is_match),
            "explanation": explanation,
        }
        results.append(result)
        print(f"        -> {explanation}")

    return results


def run(messages_path, category_description):
    messages = load_messages(messages_path)
    results = classify(messages, category_description)

    print(f"\n--- results ---")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    messages_file = sys.argv[1] if len(sys.argv) > 1 else "messages.jsonl"
    category = input("enter category description: ") if len(sys.argv) < 3 else sys.argv[2]
    run(messages_file, category)
