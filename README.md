# gmail category builder

## applicant evaluation prototype

This repository also includes a local multi-profile applicant evaluator: separate
profile centroids, source-linked capability checks, a human review queue, HTML/JSON
reports, and a synthetic comparison benchmark. See [APPLICANTS.md](APPLICANTS.md)
for setup, usage, test commands and limitations, and [benchmark results](benchmarks/RESULTS.md)
for measured behavior. Run it with `python -m applicant_eval`.

## gmail classifier

classifies gmail messages into custom categories you describe in plain english. like "work travel receipts from airlines" or "family stuff" or "newsletters about AI".

no paid APIs, everything runs locally — ollama for the LLM and sentence-transformers for embeddings.

## setup

you need:
- python 3.10+
- ollama installed and running (`brew install ollama && ollama serve`)
- llama3.2 pulled (`ollama pull llama3.2`)

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

requirements are just sentence-transformers, requests, and numpy.

## how to run

### cli (interactive)

```
python3 cli.py messages.jsonl
```

then type a category. comma-separate for multiple categories at once:

```
category> work-related travel receipts from airlines
category> family, newsletters, travel receipts
```

type `fresh` to regenerate synthetic examples. flip wrong results by typing the row number.

### direct

```
python3 classifier.py messages.jsonl "work-related travel receipts from airlines"
```

or multiple:

```
python3 classifier.py messages.jsonl "travel receipts" "family" "newsletters"
```

output is json:

```json
[
  {"message_id": "174a9", "is_in_category": true, "explanation": "Delta eTicket receipt is a travel booking confirmation from an airline"},
  {"message_id": "174b2", "is_in_category": false, "explanation": "AI Newsletter is a tech digest, not a travel receipt"}
]
```

## approach

the main problem: how do you classify emails into an arbitrary category the user just typed in?

obvious options and why i didnt use them:
- **keyword matching**: tried it first. terrible. "receipt" misses "receipts", "ai" substring matches everywhere, cant handle semantic stuff like "family"
- **per-email LLM calls**: send every email to the LLM and ask "does this match?" — works but 10k emails = 10k API calls. doesnt scale
- **embeddings with a threshold**: embed the category + emails, threshold on cosine similarity. but what threshold? 0.7? 0.8? no way to know without labeled data, which we dont have

### synthetic contrastive examples

what i ended up doing: use the LLM *once* to generate fake example emails — some that clearly match the category, some that clearly dont, and a couple tricky edge cases. then embed all of those plus the real emails with sentence-transformers (all-MiniLM-L6-v2, runs locally, 384-dim vectors). classify each real email based on whether its embedding is closer to the average of the "yes" examples or the average of the "no" examples.

basically manufacturing labeled training data from the LLMs knowledge, then using embeddings for the actual classification. the LLM understands what "work travel receipts from airlines" means and generates good examples of what matches and what doesnt. the embeddings handle the comparison at scale.

it takes 1 LLM call to generate examples regardless of how many emails you have. the embedding model is loaded once and encodes all emails in a single batch.

### why centroids and not KNN

tried KNN (k=5, each synthetic example votes individually) — actually performed worse. i think centroids average out noise from bad synthetic examples. KNN gets thrown off by individual weird ones that ollama sometimes generates. centroid is more stable with noisy data.

also tried decomposing categories into sub-questions with AND/OR/NOT logic ("work NOT meetings" = two classifiers chained). three noisy classifiers ANDed together = nothing passes. reverted.

### thresholds and edge cases

- **margin threshold (0.05)**: if the difference between yes-similarity and no-similarity is less than 0.05, we default to "no". catches borderline false positives where the model isnt really sure
- **diverse negatives**: the "no" examples are intentionally spread across different email types (shopping, social, work/calendar, newsletter, notification). early versions had narrow negatives and everything defaulted to yes
- **edge examples**: the LLM also generates 2 "tricky near-miss" examples — things that look similar to the category but shouldnt match. these get added to the no pile to tighten the decision boundary

### rule + embedding ensemble

embeddings are great for semantic matching but sometimes miss obvious surface-level stuff — like an email with "receipt" literally in the subject getting scored as borderline because the body is weird. so theres a lightweight heuristic layer on top.

ollama generates keyword rules, sender hints, and exclusion keywords for each category (one call, cached to disk alongside synthetic examples). the key design choice: rules *only* act as a tiebreaker. when the embedding margin is below 0.05 (the zone where we'd default to "no"), the rules get a vote — if they score positive, flip to yes. if not, keep the default no. rules never override a confident embedding signal.

tried two other approaches first — weighted average (broke strong embedding wins) and rules-override (newsletters about "airline travel trends" got forced to yes). tiebreaker-only was the right call.

### explanations

each result comes with a plain english explanation generated by ollama in a single batch call. the prompt includes all emails with their message IDs and classification results, and the LLM returns a JSON object keyed by ID. this means explanations are matched to emails by ID, not position — if the LLM reorders or skips one, the rest still line up correctly. falls back to positional matching (array) or generic explanations if parsing fails.

early version used separate per-email LLM calls (N calls for N emails) and a single prompt for both yes/no which confused the model. batch + ID-matching fixed both scaling and alignment.

### corrections / active learning

the cli lets you flip wrong results. corrections are saved to disk and applied as overrides on subsequent runs. tried injecting corrections back into the centroids but it broke things — correcting one airline email to "no" dragged the no-centroid toward airline content and everything airline-related became "no". direct overrides are simpler and dont have side effects.

### caching

synthetic examples are cached to disk after the first run (`synthetic_*.json`). same cache = same results every time. type `fresh` in the cli to regenerate.

## example run

```
$ python3 cli.py messages.jsonl

  gmail category builder
  12 emails loaded from messages.jsonl

  category> work-related travel receipts from airlines

   1.  YES  Your Delta eTicket Receipt                   Delta
            Delta eTicket is a work travel airline receipt
   2.   no  Weekly AI Research Digest                     AI Newsletter
            AI Newsletter is a tech digest, not a travel receipt
   3.   no  Re: Thanksgiving plans                        Mom
            Mom's Thanksgiving email is personal family communication
   4.  YES  United Airlines - Booking Confirmation         United Airlines
            United booking confirmation is a work travel airline receipt
   5.   no  Your Amazon Order Has Shipped                  Amazon
            Amazon shipping notification is online shopping, not travel
   6.   no  Invitation: Q3 Planning Meeting                Calendar
            Q3 planning invite is a work meeting, not a travel receipt
   7.   no  Papers We Love: August Edition                 PWL
            Papers We Love is a tech newsletter, not travel related
   8.   no  Re: Weekend BBQ                                Dad
            Dad's BBQ email is personal family plans
   9.  YES  American Airlines - Trip Summary               American Air...
            AA trip receipt is an airline travel receipt
  10.   no  GitHub: Pull request merged                    GitHub
            GitHub PR notification is a dev tool alert
  11.   no  Fwd: Family Reunion Photos                     Sister
            Family reunion photos are personal, not travel receipts
  12.   no  Your Spotify Wrapped is here                   Spotify
            Spotify Wrapped is a music service notification
```

note: the AA personal trip (#9) is a known false positive. the body says "personal Visa" and "personal vacation trip" but the embeddings weight the airline receipt language (flight codes, airport codes, fare) so heavily that the work vs personal distinction gets lost. the margin is well above the tiebreaker threshold so rules cant help either. this is the fundamental limitation of the embedding approach for categories where the distinguishing factor is a subtle detail in otherwise very similar content. you could fix it with per-email LLM calls but that defeats the scaling advantage.

## files

- `classifier.py` — main classification logic, synthetic example generation, embedding, rules, feedback
- `cli.py` — interactive terminal interface with corrections
- `pretty_cli.py` — aesthetic terminal interface (same classifier, nicer vibes)
- `test_classifier.py` — unit + integration tests (run with `pytest`)
- `fixtures/` — committed golden synthetic examples + rules for reproducible tests
- `messages.jsonl` — sample email dataset (12 messages)
- `requirements.txt` — python dependencies
- `dev_journal.md` — development notes and iteration history
