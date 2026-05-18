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

