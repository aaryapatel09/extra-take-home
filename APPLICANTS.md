# Applicant evidence review

A local prototype of multi-profile applicant matching. It creates a ranked **human review** report with exact source quotes, unresolved requirements, and a sampled exploration queue. It does not make hiring decisions. The Gmail classifier remains available through its original commands.

## Interactive CLI

Use Python 3.11 or later. CPU is the default and the tested backend.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Starts with fictional sample applicants:
python applicant_cli.py
# Equivalent:
python -m applicant_eval

# Start with your own records or a folder of text resumes:
python applicant_cli.py ./resumes
```

On macOS, after setup, double-click `start_applicants.command` to open the same CLI.

Type `demo` to try the included role immediately. To draft profiles from your own job description, run local Ollama and pull `llama3.2:3b` first (`ollama serve`, then `ollama pull llama3.2:3b` in another terminal). Type a job description at the prompt:

```text
job> Backend developer who can build APIs, write SQL queries, and write automated tests.
review> 2
review> mark 2 shortlist Strong project evidence; discuss testing in interview.
review> queue
review> export output/backend-review
review> quit
```

The CLI displays the draft capabilities, the exact job wording behind each, and nine example profiles across three paths: direct experience, independent projects, and transferable experience. It evaluates the loaded applicants and shows all results. Inspect the draft requirements before relying on the results. `profiles` shows them again; `role PATH` loads an edited role JSON. The saved-role format supports up to 40 paths.

| Command | Action |
|---|---|
| `demo` | Load and evaluate the fictional sample; no Ollama required |
| Plain job description / `job DESCRIPTION` | Generate or reuse cached profiles and evaluate |
| `paste` | Enter a multiline job; finish with a single `.` |
| `load PATH` | Load JSONL, a JSON array, a text resume, or a folder of text resumes |
| `role PATH` / `profiles` | Load a saved rubric / inspect its profiles |
| `run` / `fresh` | Re-evaluate / regenerate profiles for the last typed job |
| `list` / a row number | Show the table / inspect source evidence |
| `mark N shortlist\|hold\|pass [NOTE]` | Save your own review decision and optional note |
| `clear N` | Remove your review mark |
| `queue` / `budget N` | Show the review queue / change its size |
| `export [DIRECTORY]` | Save HTML, JSON, and the role, including your review notes |
| `help` / `quit` | List commands / exit |

Quote paths containing spaces. Human marks do not change model scores. Notes persist under `output/applicant-session`, scoped to the exact role and applicant text; changing either starts a distinct review. A damaged notes file is preserved and reported rather than silently overwritten. Use `--session-dir PATH` to separate projects. Review files contain applicant text and your notes.

First use downloads the pinned MiniLM embedding and DeBERTa evidence model weights from Hugging Face. Subsequent runs can use `--offline`; this means cached evaluation models, and still allows the local Ollama connection for a new job. Applicant records are evaluated locally and are not sent to the generator or an inference API. Models are loaded once per CLI session. Profile generation releases its model memory before evaluation. `--plain` disables colors and animations; `--model` and `--ollama-url` select the generator.

The local model endpoints should be left at localhost for local operation. Choosing a remote `--ollama-url` sends job descriptions to that endpoint, but never applicant records.

## Batch demo

```sh
python -m applicant_eval evaluate \
  --role examples/applicants/backend_role.json \
  --applicants examples/applicants/sample_applicants.jsonl \
  --budget 12 \
  --output output/applicants.json \
  --html output/applicants.html
```

Open `output/applicants.html` in a browser. All applicants remain visible; the review queue includes ranked candidates and a reproducible 20% sample of the remaining pool. This sample is for discovering blind spots, and can reduce immediate recall at a fixed budget. The library's `review_queue` accepts an exploration fraction and seed. A budget of one uses the first ranked record.

`requirements-applicants.lock` records the exact dependency versions used for the reported tests. To reproduce that environment, use it in place of `requirements.txt` (tested on Python 3.11, macOS ARM64 and Linux CI). Model revisions are pinned in `applicant_eval/models.py` and included in every report. CPU inference is tested; GPU device flags are provided but unverified.

## Inputs

For JSONL input, each applicant is one JSON object per line:

```json
{"id":"candidate-001","evidence":[{"id":"project-1","text":"I built a REST API for volunteer scheduling."},{"id":"project-2","text":"I wrote SQL queries and automated tests for the application."}]}
```

Supply work/project statements in `evidence`, preferably one accomplishment per entry. Full text is also supported and split into sentence-aware, overlapping token windows. Keep original evidence IDs: report offsets are zero-based, end-exclusive Python character offsets into the **unchanged input evidence text**. A quote is a self-reported claim, not independent verification of an accomplishment.

Only `id` and `evidence` are consumed. Names, contact details, school metadata, demographic attributes and benchmark labels in other fields are ignored. This does **not** remove such information if it appears inside evidence text, and does not establish fairness of embeddings or the downstream review process. Applicant IDs are used for deterministic tie-breaking and exploration sampling, not semantic scoring.

An empty evidence list is valid and produces `needs_review` with unknown criteria. Missing information never becomes an automatic rejection. Duplicate IDs, invalid role references, empty profile sets, malformed JSON, oversized records and non-finite settings fail explicitly. The interactive CLI also accepts JSON arrays and one `.txt`/`.md` file per applicant, individually or in a directory. Directory input ignores hidden files and unsupported extensions. This version does not parse PDF/DOCX résumés.

The role JSON contains:

- A job description and atomic capability criteria. Each has a short NLI hypothesis and an explicit `required` flag.
- Several qualifying paths, each with two or more example profiles and its own capability IDs. All paths must also satisfy the common required criteria.
- Optional near-miss examples. Their similarities are diagnostic, not rejection thresholds.

See the backend role fixture for production, independent-builder and research routes. The role is an illustrative rubric, not a validated job analysis.

## Optional profile generation

If local Ollama is installed and a model has been pulled:

```sh
python -m applicant_eval generate-role \
  --description job-description.txt \
  --model llama3.2:3b \
  --output output/draft-role.json
```

Generation uses two schema-constrained local steps: extract source-linked capabilities, then draft complete examples for every essential capability in three distinct experience paths. Each step allows one repair attempt, so a job takes two to four requests regardless of applicant count. Drafts are cached by job description, model, and generator version. Generated profiles contain positive examples; near-miss examples can be supplied in a reviewed role file. They are not automatically invented as rejection rules.

Validation checks structure, unique IDs, source-quote presence in the job text, complete example fields, and unrequested tenure. These checks do not establish that a draft captured every requirement or interpreted it correctly. Inspect requirements and examples before relying on a generated rubric. Live generation and the interactive flow were tested with local `llama3.2:3b`; see `benchmarks/RESULTS.md`. The checked-in demo role works without Ollama.

## Scoring and evidence checks

1. Normalize example embeddings and average **within each path**. Keep paths separate.
2. Split applicant evidence into sentence-aware token windows. Preserve offsets, deduplicate exact repeated claims, and embed every window including long-document tails.
3. Compare the applicant's mean normalized evidence vector with every path centroid. Report the strongest match. The report also includes direct-job-description, single-centroid, contrastive-centroid and strongest-chunk baselines.
4. Retrieve up to three windows for **each** capability, independent of the closest path. Run a local NLI cross-encoder over each retrieved claim and the capability hypothesis.
5. Use conservative English context flags for future plans, other people's work, requirements/instructions and explicit negation. Require minimum retrieval similarity and NLI support. NLI contradiction counts only with explicit negation; unrelated statements do not establish missing ability.
6. Mark a capability supported, unknown, contradicted or conflicting. Overall `evidence_supported` requires support for every common essential capability plus one complete path. Everyone else remains `needs_review`.
7. Rank by overall support, common capability coverage, best-path coverage, then profile similarity. Display every record and sample outside the leading ranks for exploration review.

The best-supported path can differ from the closest semantic profile. Support and similarity are **not calibrated probabilities**. The 0.8 support/contradiction thresholds and 0.25 retrieval floor are prototype heuristics, not universal hiring cutoffs. Changing role wording or thresholds requires re-evaluation. Human decisions and work-sample evidence are not simulated by these model scores.

## Testing

```sh
pip install -r requirements-applicants-dev.txt
ruff check applicant_eval applicant_cli.py tests/applicants
ruff format --check applicant_eval applicant_cli.py tests/applicants

# Fast deterministic tests, no model downloads:
pytest tests/applicants -m 'not model' -q

# Download once; omit this command when already cached:
python -c 'from applicant_eval.models import LocalModels; LocalModels()'

# Actual local model tests and coverage:
pytest tests/applicants --run-model-tests -q --cov=applicant_eval

# Existing Gmail classifier regression suite:
pytest test_classifier.py -q

# Synthetic challenge set, with all comparison methods and paired stress tests:
python -m applicant_eval benchmark \
  --role examples/applicants/backend_role.json \
  --applicants benchmarks/applicants.jsonl \
  --paraphrases benchmarks/paraphrases.jsonl \
  --split test --budget 12 --offline \
  --output output/benchmark.json --html output/benchmark.html
```

The benchmark has 12 development cases and 40 challenge cases, with 12 labeled qualified under the demo rubric. It also runs 120 paired metadata/order/duplication checks and 12 separately authored paraphrases. These pairs are correlated checks, **not additional independent applicants**. All data is invented for this project. There are no real hiring outcomes or independent reviewer labels. Test labels never enter evaluation inputs.

The initial challenge run exposed a long-tail miss; sentence-aware chunking was added and the challenge set rerun. Consequently this is a development challenge set, not an untouched external holdout. See `benchmarks/RESULTS.md` for measured results, iteration history and remaining failures. The tests check source integrity, validation, ranking behavior, context handling, malformed model outputs, sampling, repeatability, HTML escaping, long text and real-model behavior. CI runs those checks and produces benchmark artifacts.

## Limits and next validation

This is an English-language prototype for reviewer assistance. The demonstration covers one backend role. The NLI model is trained for general natural-language inference, not employment evaluation. It can misunderstand domain terms, agency, negation and legitimate paraphrases. Conservative English context flags can also flag legitimate descriptions of collaboration or leadership. Top-k retrieval can miss relevant or conflicting evidence; sentence splitting can lose cross-sentence context. Invented claims and copied experience cannot be authenticated from text alone.

Before considering real screening, evaluate on consented, independently reviewed applications across roles and use job-relevant assessments to establish outcomes. Check missed applicants at a fixed review budget, review workload, wording sensitivity and subgroup effects. Retain an auditable human route for unfamiliar or incomplete applications. The supplied tests establish software behavior and expose selected model failures; they do not establish validity for real employment decisions.
