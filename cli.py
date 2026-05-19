#!/usr/bin/env python3
"""interactive cli for the gmail category builder"""

import sys
import json
from classifier import (
    load_messages,
    classify,
    load_feedback,
    save_feedback,
    get_feedback_path,
)


def show_results(results, messages):
    msg_lookup = {m["id"]: m for m in messages}
    yes_count = sum(1 for r in results if r["is_in_category"])
    print(f"\n  {yes_count}/{len(results)} emails matched\n")
    for i, r in enumerate(results, 1):
        status = "YES" if r["is_in_category"] else " no"
        msg = msg_lookup.get(r["message_id"], {})
        subject = msg.get("subject", "???")[:50]
        print(f"  {i:2d}. [{status}] {r['message_id']}  {subject}")
        print(f"      -> {r['explanation']}")
    print()


def correction_loop(results, category_description):
    feedback = load_feedback(category_description)
    changed = 0

    print("flip any results? enter row number to toggle, 'done' when finished:")
    while True:
        choice = input("> ").strip().lower()
        if choice in ("done", "d", ""):
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                new_label = not r["is_in_category"]
                feedback[r["message_id"]] = {"corrected_label": new_label}
                status = "YES" if new_label else "no"
                print(f"  flipped {r['message_id']} -> {status}")
                changed += 1
            else:
                print(f"  pick 1-{len(results)}")
        except ValueError:
            print("  enter a number or 'done'")

    if changed > 0:
        save_feedback(category_description, feedback)
        print(f"  saved {len(feedback)} total corrections. re-run same category to apply")

    return feedback


def main():
    messages_file = sys.argv[1] if len(sys.argv) > 1 else "messages.jsonl"
    messages = load_messages(messages_file)

    print(f"\n=== gmail category builder ===")
    print(f"loaded {len(messages)} emails from {messages_file}")
    print(f"commands: type a category to classify, 'fresh' to regenerate, 'quit' to exit\n")

    last_category = None

    while True:
        category = input("category> ").strip()

        if category.lower() in ("quit", "q", "exit"):
            break
        if not category:
            continue

        fresh = False
        if category.lower() == "fresh":
            if last_category:
                category = last_category
                fresh = True
                print(f"  regenerating examples for \"{category}\"")
            else:
                print("  no previous category to regenerate")
                continue

        feedback = load_feedback(category)
        if feedback:
            print(f"  found {len(feedback)} corrections from last run")

        results = classify(messages, category, feedback if feedback else None, fresh=fresh)
        show_results(results, messages)
        last_category = category
        correction_loop(results, category)


if __name__ == "__main__":
    main()
