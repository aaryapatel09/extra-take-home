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

