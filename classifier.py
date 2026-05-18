#!/usr/bin/env python3
"""
gmail custom category builder - v0 (keyword matching, intentionally naive)
this is the dumb version. just splits the category description into words
and checks if any of them appear in the email. gonna be bad but it gets
the pipeline working end to end.
"""

import json
import base64
import sys
import re


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
        # strip html tags if present - super naive regex approach
        # TODO: this is gonna break on nested tags, malformed html, etc
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    except Exception as e:
        return f"[decode error: {e}]"


def extract_keywords(category_description):
    """pull 'meaningful' words from the category description.
    just lowercases and removes stopwords. this is the dumb part."""
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "about", "above", "after", "again", "against", "all", "am",
        "and", "any", "are", "as", "at", "because", "before", "between",
        "both", "but", "by", "down", "during", "each", "few", "for",
        "from", "further", "get", "got", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
        "once", "only", "or", "other", "our", "ours", "ourselves", "out",
        "over", "own", "re", "same", "she", "so", "some", "such", "than",
        "that", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "we", "what", "when", "where", "which",
        "while", "who", "whom", "why", "with", "you", "your", "yours",
        "related",
    }
    words = re.findall(r"\w+", category_description.lower())
    keywords = [w for w in words if w not in stopwords]
    return keywords


def classify_message(msg, keywords):
    """check if any keyword appears in subject, snippet, from, or decoded body.
    returns (is_match, explanation)"""
    body_text = decode_body(msg.get("body", ""))
    searchable = " ".join([
        msg.get("subject", ""),
        msg.get("snippet", ""),
        msg.get("from", ""),
        body_text,
    ]).lower()

    matched = [kw for kw in keywords if kw in searchable]

    if matched:
        return True, f"matched keywords: {', '.join(matched)}"
    else:
        return False, "no keyword matches found"


def run(messages_path, category_description):
    messages = load_messages(messages_path)
    keywords = extract_keywords(category_description)

    print(f"\n--- category: \"{category_description}\"")
    print(f"--- extracted keywords: {keywords}")
    print(f"--- processing {len(messages)} messages\n")

    results = []
    for msg in messages:
        is_match, explanation = classify_message(msg, keywords)
        result = {
            "message_id": msg["id"],
            "is_in_category": is_match,
            "explanation": explanation,
        }
        results.append(result)

        status = "YES" if is_match else " no"
        print(f"  [{status}] {msg['id']}: {msg['subject'][:50]}")

    print(f"\n--- results ---")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    messages_file = sys.argv[1] if len(sys.argv) > 1 else "messages.jsonl"
    category = input("enter category description: ") if len(sys.argv) < 3 else sys.argv[2]
    run(messages_file, category)
