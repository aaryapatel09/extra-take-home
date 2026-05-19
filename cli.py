#!/usr/bin/env python3
"""interactive cli for the gmail category builder"""

import sys
import json
from classifier import (
    load_messages,
    classify,
    classify_multi,
    load_feedback,
    save_feedback,
    get_feedback_path,
)

BOLD = "\033[1m"
DIM = "\033[90m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"
LINE = f"{DIM}{'─' * 74}{RESET}"


def show_results(results, messages, category=None):
    msg_lookup = {m["id"]: m for m in messages}
    yes_count = sum(1 for r in results if r["is_in_category"])

    if category:
        print(f"\n  {BOLD}{category}{RESET}  {DIM}|{RESET}  {yes_count}/{len(results)} matched")
    else:
        print(f"\n  {yes_count}/{len(results)} matched")
    print(f"  {LINE}")

    for i, r in enumerate(results, 1):
        msg = msg_lookup.get(r["message_id"], {})
        subject = msg.get("subject", "???")[:44]
        sender = msg.get("from", "").split("<")[0].strip()[:16]

        if r["is_in_category"]:
            tag = f"{GREEN}{BOLD} YES {RESET}"
        else:
            tag = f"{DIM}  no {RESET}"

        explanation = r["explanation"][:60]
        print(f"  {i:2d}. {tag} {subject:<44s} {DIM}{sender}{RESET}")
        print(f"           {DIM}{explanation}{RESET}")

    print()


def correction_loop(results, category_description):
    feedback = load_feedback(category_description)
    changed = 0

    print(f"  {DIM}flip a result? enter row number, or press enter to continue{RESET}")
    while True:
        try:
            choice = input(f"  {DIM}>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice in ("done", "d", ""):
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                new_label = not r["is_in_category"]
                feedback[r["message_id"]] = {"corrected_label": new_label}
                tag = f"{GREEN}YES{RESET}" if new_label else f"{DIM}no{RESET}"
                print(f"    flipped {r['message_id']} -> {tag}")
                changed += 1
            else:
                print(f"    {DIM}pick 1-{len(results)}{RESET}")
        except ValueError:
            print(f"    {DIM}enter a number or press enter{RESET}")

    if changed > 0:
        save_feedback(category_description, feedback)
        print(f"  {DIM}saved {len(feedback)} corrections{RESET}\n")

    return feedback


def main():
    messages_file = sys.argv[1] if len(sys.argv) > 1 else "messages.jsonl"
    messages = load_messages(messages_file)

    print(f"\n  {BOLD}gmail category builder{RESET}")
    print(f"  {DIM}{len(messages)} emails loaded from {messages_file}{RESET}")
    print(f"  {DIM}comma-separate for multi | 'fresh' to regenerate | 'quit' to exit{RESET}\n")

    last_categories = None

    while True:
        try:
            raw = input(f"  {CYAN}category>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw.lower() in ("quit", "q", "exit"):
            break
        if not raw:
            continue

        fresh = False
        if raw.lower() == "fresh":
            if last_categories:
                categories = last_categories
                fresh = True
                print(f"  {DIM}regenerating examples...{RESET}")
            else:
                print(f"  {DIM}no previous category{RESET}")
                continue
        else:
            categories = [c.strip() for c in raw.split(",") if c.strip()]

        if len(categories) == 1:
            category = categories[0]
            feedback = load_feedback(category)
            if feedback:
                print(f"  {DIM}{len(feedback)} corrections loaded{RESET}")

            results = classify(messages, category, feedback if feedback else None, fresh=fresh)
            show_results(results, messages)
            last_categories = categories
            correction_loop(results, category)
        else:
            feedbacks = {}
            for cat in categories:
                fb = load_feedback(cat)
                if fb:
                    feedbacks[cat] = fb
                    print(f"  {DIM}{len(fb)} corrections for \"{cat}\"{RESET}")

            all_results = classify_multi(messages, categories, feedbacks, fresh=fresh)

            for cat, results in all_results.items():
                show_results(results, messages, category=cat)

            last_categories = categories

            for cat, results in all_results.items():
                print(f"  {BOLD}{cat}{RESET} corrections:")
                correction_loop(results, cat)

    print(f"\n  {DIM}done{RESET}\n")


if __name__ == "__main__":
    main()
