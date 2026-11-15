# Applicant evaluation: test results

Measured on 2026-11-15 with Python 3.11.16 on macOS ARM64, CPU inference.

## Scope

- **138 applicant-evaluator tests passed**, including four tests using actual local models.
- **101 existing Gmail classifier tests passed**.
- **93.8% statement coverage** for the applicant package.
- **52 authored synthetic applicants**: 12 development cases and 40 challenge cases. The challenge set contains 12 records labeled qualified under the demo rubric and 28 with insufficient or conflicting evidence.
- **120 paired invariance checks**: 40 changes to identity metadata, 40 reorderings and 40 exact-evidence duplications.
- **12 authored paraphrase pairs**, evaluated separately.
- Ruff lint and formatting checks passed. GitHub CI runs are reported separately from these local results.

These are developer-authored labels for one illustrative backend role, not independent hiring judgments or observed job performance. The challenge set intentionally contains difficult negatives; it does not represent applicant prevalence in a real hiring funnel. Paired cases are correlated, not additional independent applicants.

## Fixed review budget comparison

Every method ranks the same 40 applicants. Review budget: 12. All baselines use the same pinned embedding model and source preprocessing. Contrastive methods use the fixture's near-miss profiles as negatives. No test label enters the evaluator.

| Method | Qualified found | Recall at 12 | Average precision |
|---|---:|---:|---:|
| Job-description similarity | 1/12 | 8.3% | 0.240 |
| Single positive centroid | 6/12 | 50.0% | 0.506 |
| Single positive-minus-negative centroid | 9/12 | 75.0% | 0.789 |
| Multiple positive centroids | 8/12 | 66.7% | 0.648 |
| Multiple contrastive profiles | 9/12 | 75.0% | 0.810 |
| Strongest matching chunk | 2/12 | 16.7% | 0.354 |
| Evidence-based ranking | 11/12 | 91.7% | 0.994 |
| Default review queue with 20% exploration | 9/12 | 75.0% | 0.946 |

The evidence ranking found **11 of 12** qualified examples in its first 12 positions. The default queue found **9 of 12**, because three slots are reserved for exploring lower-ranked applicants. This is an explicit tradeoff, not an improvement in immediate recall. All 40 applicants remain visible in the report.

## Observed misses and false support

- **0 of 28** insufficient-evidence base cases received the overall `evidence_supported` label in this run. This does not establish a population false-positive rate or guarantee every individual capability assessment is correct.
- **9 of 12** qualified base cases received that label. The remaining three still require review because the NLI model did not establish their production-operation criterion.
- `t01-payments` ranks **13th**, outside the 12-person budget. Its implementation, SQL and testing evidence was recognized; its production-service wording was not.
- `t02-inventory` and `t08-maintainer` remain `needs_review` but appear within the ranked top 12.
- The familiar-looking but incomplete `t24-no-route` enters the top 12; its unresolved route requirement is explicitly shown.

## Paired robustness

All **120** metadata/order/duplication pairs retained identical capability outcomes and profile similarity scores. Identity fields are excluded from scoring by construction; this is not a demographic fairness audit, and identity-related information embedded in work text is not automatically removed.

Of **12 paraphrase pairs**, **7** changed a capability assessment or best-supported path, and **1** changed the overall review label. `rewrite-t11-consultant` changed from `evidence_supported` to `needs_review` despite preserving the facts. No negative paraphrase received overall evidence support. Capability-level errors still occur; a reviewer must inspect the cited passages.

## What testing changed

The first development run incorrectly credited a recruiter with engineers' work and interpreted unrelated statements as contradictions. Context flags, a relevance floor, and explicit-negation requirements were added using those development cases. The first challenge run found 11/12 qualified applicants, missed the long-tail candidate, and had average precision 0.953. Sentence-aware token windows recovered the long-tail claim; the rerun retained 11/12 recall but improved average precision to 0.994. Paraphrase testing then exposed a false negation flag for “not only”; that parsing issue was fixed and rerun. The results therefore describe an iterated challenge suite, **not an untouched external holdout**.

## Reproducibility and limits

Model revisions, settings, dataset digest, comparisons, slices and paired failures are recorded in `benchmarks/results.json`. Full source-linked outputs are produced by the command in `APPLICANTS.md`. Exact tested packages are listed in `requirements-applicants.lock`.

The last benchmark took approximately **4.7 seconds after model loading**, including paired checks. In-process caching reuses repeated text; this is not a production throughput claim.

The interactive CLI was tested in a real terminal with the actual embedding/evidence models and a local Ollama `llama3.2:3b` model. Live drafting captured the three requested capabilities for both a backend job and a frontend job, using two generation requests per job (approximately 35–41 seconds on this machine). These were generation smoke tests, not applicant-quality benchmarks for a second role.

The first live drafts exposed duplicate IDs, invented tenure, incomplete skills, and negative examples based on optional requirements. Generation now separates source-linked capability extraction from example generation, validates every essential example field, and permits one bounded repair per step. Generated near misses were removed; reviewed role files can still provide them. Exact job quotes and draft examples remain inspectable. This improves structural reliability but does not guarantee semantic completeness: generated examples still sometimes describe only loosely related work for a capability.

The CLI tests cover cached roles, single-session model reuse, persisted human notes, isolation when job/applicant contents change, malformed inputs, generation failures, incomplete examples, escaped display/export, and graceful EOF/interruption. The manual terminal walkthrough also checked generation, evidence inspection, marking, queue size, exports, and cache reuse. PDF parsing, other languages, GPU evaluation and real hiring validity were not tested.

For real-world validation, the next evidence needed is a consented corpus independently assessed against job-relevant tasks, including audits of applicants below the review cutoff. These results support further experimentation with reviewer assistance; they do not justify automatic applicant rejection.
