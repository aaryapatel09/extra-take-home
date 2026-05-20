# dev journal

## starting out

build a gmail category builder. user types smth like "work travel receipts from airlines" and it classifies emails

approaches:
- keyword matching — fast but terrible
- embeddings — threshhold problem, 0.7? 0.8? unsolvable w/o labeled data
- per-email LLM — 10k emails = 10k api calls no thx
- hybrid — overkill

### v0 keyword matching
got pipeline working. load jsonl decode base64 match keywords output json

results sucked:
- missed delta bc "receipt" vs "receipts"
- "ai" substring matches evrywhere
- missed mom/dad for "family"

---

## synthetic contrastive idea

what if you use the LLM once to genearte fake emails that match and dont match. then embed those + real emails and classify based on which pile each one is closer to

manufacturing labeled data from nothing. ollama w llama3.2 locally, sentence-transformers for embedings

first bug was json parsing. asked for 15 examples output got truncated mid-object. cut to 10 added regex fallback

### v1 results
- travel 10/12
- newsletters ~6/12 tons of false positives
- family 5/12 bad evrything defaulted yes

the no examples are too narrow so evrything looks more like yes

---

## diverse negatives

told LLM to make negatives span diff types — shopping social work/calendar newsletter

newsletters went from ~6/12 to 12/12 just from that one prompt change. family 5/12 to ~8/12

also added MARGIN_THRESHOLD = 0.05. below that default to no. catches borderline false positives

results: 11/12, 12/12, 11/12 across categories

noticed results change between runs tho. ollama generates diff examples each time. same code same data diff results

AA personal trip still wrong — embeddings cant tell work trip from personal trip. margin is 0.2+ so threshold cant help

---

## ok trying KNN + pipeline

maybe KNN k=5 instead of centroids? individual examples vote instead of averaging into one point

also tried breaking category into sub-questions w AND/OR/NOT logic so "work NOT meetings" becomes two classifiers chained

pipeline: LLM decomposed into 3 steps, three noisy classifiers ANDed = nothing passes. travel went 0/12. had to force 2 steps. then LLM hallucinated negation where there shoudlnt be one

KNN: actually worse than centroids on most runs?? i think centroids average out noise from bad synthetic examples. KNN gets thrown off by individual bad ones

tried multi-round generation too generate twice and combine. nope second round contradicts the first

reverting evrything. centroid + diverse negatives + threshold is the most stable combo. left the KNN code in but disabled it

---

## explanations

added second LLM call per email for plain english explanations. instead of similarity scores you get "matches bc its from Delta about a travel receipt"

bug: LLM said "does not match" for an email classified yes. split into two prompts one for matches one for non-matches. fixed it

tradeoff: 1 extra LLM call per email

---

## active learning + caching + cli

tried adding corrected emails to centroids for active learning. broke evrything — correcting AA trip to no dragged no centroid toward airline content, delta and united both became no. switched to direct overrides after classification

caching: just load synthetic examples from disk if they exist instead of regenerating. same cache = same results. type fresh to regenerate

cli: interactive loop enter category see results flip wrong ones re-run

---

## batch embedding + edge examples

was encoding emails one at a time in a loop. sentence-transformers can take a list and batch it on gpu/cpu way faster. switched to encode all emails in one call

also realized edge examples (the tricky near-misses) were being generated but never used. added them to the no pile — they tighten the decision boundary since theyre specifically designed to look like matches but arent

---

## multiple categories

added classify_multi — pass several categories at once like "work receipts, family, newsletters". loads embedding model once, batch embeds emails once, then runs each category against the same embeddings. only the synthetic example generation + centroid computation happens per-category

cli supports comma-separated input. corrections still work per category

---

## rule + embedding ensemble

embeddings are great for semantic matching but sometimes they miss obvious surface-level stuff. like if the category is "receipts" and the email literally has "receipt" in the subject, the embeddings might still score it borderline if the rest of the email is weird. figured i should combine simple keyword/domain heuristics with the embedding classifier

### v1: weighted average (bad)

first attempt was dumb. generated keyword rules from ollama (same as synthetic examples — one extra call, cached), then combined rule score with embedding similarity as a weighted average. 70% embedding + 30% rules

broke immediately. an email from delta about a flight change had strong embedding match but the rules didnt find "receipt" in the body so the weighted average dragged it below threshold. clear embedding wins were getting overridden by rules that happened to miss. the whole point of embeddings is semantic understanding and i was undermining that with keyword matching

### v2: rules override embeddings (also bad)

tried the opposite — if rules strongly match (3+ keyword hits), override the embedding result. this was even worse. a newsletter about "airline industry travel trends" had "airline", "travel", "receipt" mentioned in passing and the rules forced it to yes. the embeddings correctly said no because the content was newsletter-y not receipt-y, but the rules overruled

### v3: tiebreaker only (good)

realized the right approach: rules should only speak when embeddings cant decide. specifically, when the embedding margin is below 0.05 (the threshold where we default to no), let the rules break the tie instead. if rules score positive, flip to yes. if not, keep the default no

this means rules never override a strong embedding signal. they only help in the gray zone where the embeddings are genuinely unsure. tested and it caught a borderline email that embeddings couldnt quite place but had obvious keyword matches

also had a bug where the LLM generated overlapping keywords — "travel" appeared in both keywords and exclude_keywords for "work travel receipts" because it included "travel blog" as an exclusion pattern. had to deduplicate by removing any keyword that also appears in the exclude list

rules are cached alongside synthetic examples. one extra ollama call per category, cached to disk after first run

---

## hardening pass

went back through everything and fixed a bunch of stuff that was bugging me

### print → logging

classifier.py used print() for everything — progress messages, errors, debug info. pretty_cli.py had this gross hack where it redirected sys.stdout to io.StringIO during classify() to hide the prints from the spinner. if json parsing failed the error message literally vanished into the void

switched everything to `logging.getLogger(__name__)`. cli.py configures logging at INFO level so you see progress. pretty_cli.py sets classifier logger to WARNING so the spinner runs clean. no more stdout redirect

### sys.exit in library code

call_ollama had sys.exit(1) on connection error. fine for cli mode but if you import classify() as a library it kills your whole process. replaced with OllamaConnectionError exception. both CLIs catch it and show the error nicely

### explanation batch was fragile

the batch explanation prompt returned a JSON array and we matched by position. if the LLM returned fewer items or reordered them, explanations would silently attach to the wrong emails. happened once during testing — LLM skipped an explanation and every subsequent email got the wrong one

switched to ID-keyed output: prompt includes message IDs, LLM returns `{"174a9": "explanation", ...}`, we match by key. falls back to positional if LLM returns an array instead. and falls back to generic if everything fails

### committed fixtures for reproducibility

the synthetic examples and rules were gitignored (they're runtime cache). which meant tests couldnt assert classification accuracy — they could only check the plumbing (right keys, right types). a new clone couldnt reproduce the sample run results either

created fixtures/ with golden synthetic examples and rules for the travel receipts category. hand-curated — diverse senders, realistic subjects, no duplicates. test suite loads these fixtures + real messages.jsonl + real embedding model and asserts Delta=yes, United=yes, Mom=no, GitHub=no, etc. if someone breaks classification quality the test catches it

also added fixture quality assertions — checks that yes examples mention airline content, no examples span at least 4 different types (shopping, family, work, newsletter, notification), all senders unique

### test count 77 → 101

new tests: explanation ID-matching (3), reorder tolerance, array fallback, empty messages, tiebreaker through full classify pipeline with controlled embeddings, OllamaConnectionError (connection, HTTP, bad JSON), timeout verification, fixture existence, fixture quality validation, end-to-end classification accuracy, single-message classify, result ID matching, explanation count matching, JSON serialization roundtrip, multi-category with single category, score_rules edge cases (missing body, case insensitivity), parse_json edge cases (empty dict, empty array, unicode), decode_body edge cases (large content, mixed HTML/text)
