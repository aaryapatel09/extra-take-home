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
