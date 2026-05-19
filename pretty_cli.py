#!/usr/bin/env python3
"""the pretty version of the cli. same classifier, nicer vibes."""

import sys
import os
import re
import io
import time
import threading
from classifier import (
    load_messages,
    classify,
    classify_multi,
    load_feedback,
    save_feedback,
)


# ── color palette ──────────────────────────────────────────

BLUSH      = "\033[38;5;218m"
SAKURA     = "\033[38;5;213m"
ROSE       = "\033[38;5;174m"
CREAM      = "\033[38;5;223m"
LAVENDER   = "\033[38;5;183m"
MUTED      = "\033[38;5;181m"
SOFT_WHITE = "\033[38;5;255m"
WARM_WHITE = "\033[38;5;224m"
DIM        = "\033[38;5;243m"
DEEP_ROSE  = "\033[38;5;168m"
PEACH      = "\033[38;5;216m"
MAUVE      = "\033[38;5;139m"
BOLD       = "\033[1m"
ITALIC     = "\033[3m"
RESET      = "\033[0m"

GRAD = [
    "\033[38;5;225m",
    "\033[38;5;218m",
    "\033[38;5;213m",
    "\033[38;5;212m",
    "\033[38;5;211m",
    "\033[38;5;175m",
    "\033[38;5;174m",
    "\033[38;5;138m",
]


# ── helpers ────────────────────────────────────────────────

def strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*m', '', text)


def visible_len(text):
    return len(strip_ansi(text))


def pad_right(text, width):
    vl = visible_len(text)
    return text + " " * max(0, width - vl)


def gradient_text(text, colors=None):
    if colors is None:
        colors = GRAD
    out = []
    ci = 0
    for ch in text:
        if ch == " ":
            out.append(ch)
        else:
            out.append(f"{colors[ci % len(colors)]}{ch}")
            ci += 1
    out.append(RESET)
    return "".join(out)


# ── decorative tokens ─────────────────────────────────────

W = 66

FLOWER   = f"{BLUSH}✿{RESET}"
PETAL    = f"{SAKURA}❀{RESET}"
SPARKLE  = f"{LAVENDER}✧{RESET}"
HEART    = f"{ROSE}♡{RESET}"
DOT      = f"{MUTED}·{RESET}"
LEAF     = f"{MAUVE}❧{RESET}"


def divider(style="floral"):
    if style == "floral":
        segment = f"{MUTED}─────{RESET}"
        return f"  {segment} {SPARKLE} {segment} {FLOWER} {segment} {SPARKLE} {segment}"
    elif style == "thin":
        return f"  {MUTED}{'─' * W}{RESET}"
    elif style == "dots":
        dots = f" {DOT} ".join([""] * 20)
        return f"  {MUTED}{dots}{RESET}"
    elif style == "fade":
        parts = []
        chars = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        for i, c in enumerate(chars):
            shade = 240 - (i * 4 if i < len(chars)//2 else (len(chars)-1-i) * 4)
            shade = max(232, min(255, shade))
            parts.append(f"\033[38;5;{shade}m{c}")
        return f"    {''.join(parts)}{RESET}"


def box_line(text, width=W, border_color=ROSE):
    inner = width - 4
    padded = pad_right(text, inner)
    return f"  {border_color}│{RESET} {padded} {border_color}│{RESET}"


def box_top(width=W, border_color=ROSE):
    return f"  {border_color}╭{'─' * (width - 2)}╮{RESET}"


def box_bottom(width=W, border_color=ROSE):
    return f"  {border_color}╰{'─' * (width - 2)}╯{RESET}"


def box_empty(width=W, border_color=ROSE):
    return box_line("", width, border_color)


# ── spinner ────────────────────────────────────────────────

class Spinner:
    def __init__(self, message=""):
        self.message = message
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while self.running:
            frame = f"{SAKURA}{frames[i % len(frames)]}{RESET}"
            sys.stdout.write(f"\r  {frame} {MUTED}{self.message}{RESET}  ")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def stop(self, done_msg=None):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * 80 + "\r")
        if done_msg:
            print(f"  {SPARKLE} {MUTED}{done_msg}{RESET}")


# ── screens ────────────────────────────────────────────────

LOGO = [
    "                  _ _",
    "  _ __ ___   __ _(_) |",
    " | '_ ` _ \\ / _` | | |",
    " | | | | | | (_| | | |",
    " |_| |_| |_|\\__,_|_|_|",
]

def welcome_screen(email_count, filename):
    os.system("")
    print()

    bc = ROSE

    print(box_top(border_color=bc))
    print(box_empty(border_color=bc))

    for i, line in enumerate(LOGO):
        color_line = gradient_text(line, [SAKURA, BLUSH, ROSE, DEEP_ROSE, MAUVE])
        print(box_line(f"  {BOLD}{color_line}", border_color=bc))

    print(box_empty(border_color=bc))
    print(box_line(f"         {FLOWER}  {BOLD}{CREAM}gmail category builder{RESET}  {FLOWER}", border_color=bc))
    print(box_empty(border_color=bc))
    print(box_line(f"      {MUTED}{ITALIC}classify your emails with natural language{RESET}", border_color=bc))
    print(box_empty(border_color=bc))

    bar = gradient_text("━" * 40, [MUTED, ROSE, SAKURA, BLUSH, SAKURA, ROSE, MUTED])
    print(box_line(f"       {bar}", border_color=bc))
    print(box_empty(border_color=bc))

    print(box_line(f"       {SPARKLE}  {WARM_WHITE}{email_count} emails{RESET} {MUTED}loaded from{RESET} {CREAM}{filename}{RESET}", border_color=bc))
    print(box_empty(border_color=bc))
    print(box_bottom(border_color=bc))

    print()
    print(f"    {MUTED}type a category to classify    {DOT}  comma-separate for multiple{RESET}")
    print(f"    {MUTED}type{RESET} {CREAM}fresh{RESET} {MUTED}to regenerate         {DOT}  type{RESET} {CREAM}quit{RESET} {MUTED}to exit{RESET}")
    print()
    print(divider("fade"))
    print()


def show_results(results, messages, category=None):
    msg_lookup = {m["id"]: m for m in messages}
    yes_count = sum(1 for r in results if r["is_in_category"])
    no_count = len(results) - yes_count

    print()

    if category:
        cat_display = gradient_text(category, [BLUSH, SAKURA, ROSE])
        print(f"    {FLOWER}  {BOLD}{cat_display}")
    print()

    bar_width = 30
    if len(results) > 0:
        yes_bar = round((yes_count / len(results)) * bar_width)
        no_bar = bar_width - yes_bar
        bar = f"{SAKURA}{'█' * yes_bar}{RESET}{MUTED}{'░' * no_bar}{RESET}"
        print(f"    {bar}  {SAKURA}{yes_count} matched{RESET} {MUTED}· {no_count} skipped{RESET}")
    print()
    print(divider("floral"))
    print()

    for i, r in enumerate(results, 1):
        msg = msg_lookup.get(r["message_id"], {})
        subject = msg.get("subject", "???")[:44]
        sender = msg.get("from", "").split("<")[0].strip()[:18]

        if r["is_in_category"]:
            tag = f"{BOLD}{SAKURA}yes{RESET}"
            icon = PETAL
            subj_style = f"{CREAM}"
            border = f"{SAKURA}┃{RESET}"
        else:
            tag = f"{DIM}no{RESET}"
            icon = DOT
            subj_style = f"{DIM}"
            border = f"{DIM}┃{RESET}"

        explanation = r["explanation"]

        num = f"{DIM}{i:2d}{RESET}"
        print(f"    {border} {num}  {icon} {tag:<20s}  {subj_style}{subject}{RESET}")
        print(f"    {border}             {DIM}{explanation}{RESET}")
        print(f"    {border}             {DIM}from {sender}{RESET}")

        if i < len(results):
            print(f"    {DIM}·{RESET}")

    print()
    print(divider("fade"))
    print()


def correction_loop(results, category_description):
    feedback = load_feedback(category_description)
    changed = 0

    print(f"    {MUTED}flip a result? enter row #, or press enter to continue{RESET}")
    print()

    while True:
        try:
            choice = input(f"    {ROSE}❀{RESET} {DIM}›{RESET} ").strip().lower()
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
                if new_label:
                    print(f"      {PETAL} flipped #{idx+1} to {BOLD}{SAKURA}yes{RESET}")
                else:
                    print(f"      {DOT} flipped #{idx+1} to {DIM}no{RESET}")
                changed += 1
            else:
                print(f"      {MUTED}pick 1–{len(results)}{RESET}")
        except ValueError:
            print(f"      {MUTED}enter a number or press enter{RESET}")

    if changed > 0:
        save_feedback(category_description, feedback)
        print(f"\n    {SPARKLE} {MUTED}saved {len(feedback)} corrections{RESET}\n")

    return feedback


def goodbye():
    print()
    bc = ROSE
    print(box_top(width=40, border_color=bc))
    print(box_empty(width=40, border_color=bc))
    msg = f"{FLOWER}  {CREAM}see you next time{RESET}  {FLOWER}"
    print(box_line(msg, width=40, border_color=bc))
    print(box_empty(width=40, border_color=bc))
    print(box_bottom(width=40, border_color=bc))
    print()


# ── main loop ──────────────────────────────────────────────

def main():
    messages_file = sys.argv[1] if len(sys.argv) > 1 else "messages.jsonl"
    messages = load_messages(messages_file)

    welcome_screen(len(messages), messages_file)

    last_categories = None

    while True:
        try:
            raw = input(f"    {BLUSH}✿{RESET} {CREAM}category{RESET} {ROSE}›{RESET}  ").strip()
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
            else:
                print(f"\n    {MUTED}no previous category{RESET}\n")
                continue
        else:
            categories = [c.strip() for c in raw.split(",") if c.strip()]

        print()
        spinner = Spinner("classifying your emails")
        spinner.start()

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        if len(categories) == 1:
            category = categories[0]
            fb = load_feedback(category)
            results = classify(messages, category, fb if fb else None, fresh=fresh)
            sys.stdout = old_stdout
            spinner.stop(f"classified {len(results)} emails")
            show_results(results, messages)
            last_categories = categories
            correction_loop(results, category)
        else:
            feedbacks = {}
            for cat in categories:
                fb = load_feedback(cat)
                if fb:
                    feedbacks[cat] = fb

            all_results = classify_multi(messages, categories, feedbacks, fresh=fresh)
            sys.stdout = old_stdout
            spinner.stop(f"classified across {len(categories)} categories")

            for cat, results in all_results.items():
                show_results(results, messages, category=cat)

            last_categories = categories

            for cat, results in all_results.items():
                cat_display = gradient_text(cat, [BLUSH, SAKURA, ROSE])
                print(f"    {PETAL}  {BOLD}{cat_display}{RESET} {MUTED}corrections:{RESET}")
                correction_loop(results, cat)

    goodbye()


if __name__ == "__main__":
    main()
