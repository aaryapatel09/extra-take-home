"""A reproducible synthetic challenge benchmark, not a validated hiring assessment."""

import hashlib
import json
import time
from copy import deepcopy

from .engine import evaluate, review_queue
from .schema import Applicant


def metrics(order, labels, k):
    k = min(k, len(order))
    positives = sum(labels.values())
    hits = sum(labels[i] for i in order[:k])
    seen = 0
    ap = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            seen += 1
            ap += seen / rank
    return {
        "review_budget": k,
        "qualified_found": hits,
        "qualified_total": positives,
        "precision_at_k": hits / k if k else 0.0,
        "recall_at_k": hits / positives if positives else 0.0,
        "average_precision": ap / positives if positives else 0.0,
        "missed_qualified_ids": [i for i in order[k:] if labels[i]],
    }


def behavioral_signature(result):
    return (
        result["review_status"],
        result["best_supported_profile"],
        tuple(sorted((c["criterion_id"], c["status"]) for c in result["evidence"])),
    )


def run_benchmark(role, records, models, settings, budget=12, split="test", paraphrases=None):
    records = [r for r in records if r.get("split") == split]
    if not records or any(type(r.get("expected_qualified")) is not bool for r in records):
        raise ValueError(
            "benchmark requires nonempty records with boolean expected_qualified labels"
        )
    labels = {r["id"]: r["expected_qualified"] for r in records}
    started = time.perf_counter()
    report = evaluate(role, [Applicant.from_dict(r) for r in records], models, settings)
    results = report["results"]
    comparisons = {}
    for method in (
        "job_description",
        "single_centroid",
        "contrastive_single_centroid",
        "multiple_centroids",
        "contrastive_multiple_centroids",
        "chunk_matching",
    ):
        order = [
            r["applicant_id"]
            for r in sorted(results, key=lambda r: (-r["baselines"][method], r["applicant_id"]))
        ]
        comparisons[method] = metrics(order, labels, budget)
    order = [r["applicant_id"] for r in results]
    comparisons["evidence_review"] = metrics(order, labels, budget)
    queue = review_queue(results, budget)
    selected = [q["applicant_id"] for q in queue]
    comparisons["evidence_review_with_exploration"] = metrics(
        selected + [i for i in order if i not in selected], labels, budget
    )
    by_id = {r["applicant_id"]: r for r in results}
    support_errors = [
        r["applicant_id"]
        for r in results
        if r["review_status"] == "evidence_supported" and not labels[r["applicant_id"]]
    ]
    support_misses = [
        r["applicant_id"]
        for r in results
        if r["review_status"] != "evidence_supported" and labels[r["applicant_id"]]
    ]

    # Paired stress cases are reported separately, never counted as independent applicants.
    variants = []
    for record in records:
        for variant in ("identity_metadata", "evidence_order", "duplicate_evidence"):
            v = deepcopy(record)
            v["id"] = record["id"] + "::" + variant
            if variant == "identity_metadata":
                v.update(
                    name="Different identity",
                    age=61,
                    gender="woman",
                    school="Different school",
                    disability="metadata only",
                    expected_qualified=not record["expected_qualified"],
                )
            elif variant == "evidence_order":
                v["evidence"].reverse()
            elif v["evidence"]:
                v["evidence"] += [
                    {"id": e["id"] + "-duplicate", "text": e["text"]} for e in v["evidence"]
                ]
            variants.append(v)
    variant_report = evaluate(role, [Applicant.from_dict(v) for v in variants], models, settings)
    invariance = {
        v: {"pairs": len(records), "status_changes": [], "max_similarity_change": 0.0}
        for v in ("identity_metadata", "evidence_order", "duplicate_evidence")
    }
    for r in variant_report["results"]:
        original, variant = r["applicant_id"].rsplit("::", 1)
        stat = invariance[variant]
        if behavioral_signature(r) != behavioral_signature(by_id[original]):
            stat["status_changes"].append(original)
        stat["max_similarity_change"] = max(
            stat["max_similarity_change"],
            abs(r["profile_similarity"] - by_id[original]["profile_similarity"]),
        )
    slices = {}
    for tag in sorted({t for r in records for t in r.get("tags", [])}):
        ids = {r["id"] for r in records if tag in r.get("tags", [])}
        slices[tag] = {
            "count": len(ids),
            "qualified": sum(labels[i] for i in ids),
            "qualified_found": sum(labels[i] for i in order[:budget] if i in ids),
            "false_supported": sorted(ids.intersection(support_errors)),
        }
    report["review_queue"] = queue
    paraphrase_checks = {
        "pairs": 0,
        "status_changes": [],
        "false_supported_ids": [],
        "qualified_needing_review_ids": [],
    }
    if paraphrases:
        if any(p.get("variant_of") not in by_id for p in paraphrases):
            raise ValueError("paraphrase references an applicant outside the selected split")
        links = {p["id"]: p["variant_of"] for p in paraphrases}
        paired = evaluate(role, [Applicant.from_dict(p) for p in paraphrases], models, settings)
        paraphrase_checks["pairs"] = len(paraphrases)
        for r in paired["results"]:
            original = links[r["applicant_id"]]
            if behavioral_signature(r) != behavioral_signature(by_id[original]):
                paraphrase_checks["status_changes"].append(
                    {
                        "variant": r["applicant_id"],
                        "original": original,
                        "original_status": by_id[original]["review_status"],
                        "variant_status": r["review_status"],
                        "changed_criteria": [
                            c["criterion_id"]
                            for c, o in zip(r["evidence"], by_id[original]["evidence"])
                            if c["status"] != o["status"]
                        ],
                    }
                )
            if r["review_status"] == "evidence_supported" and not labels[original]:
                paraphrase_checks["false_supported_ids"].append(r["applicant_id"])
            if r["review_status"] != "evidence_supported" and labels[original]:
                paraphrase_checks["qualified_needing_review_ids"].append(r["applicant_id"])
        paraphrase_checks["evaluation"] = paired
    return {
        "dataset": "authored synthetic challenge set",
        "split": split,
        "independent_records": len(records),
        "paired_variants": len(variants),
        "dataset_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "limitations": [
            "Labels are author judgments under the fixture rubric, not hiring outcomes.",
            "No external or prospective validation; model scores are uncalibrated.",
            "Metadata invariance is enforced by input exclusion, not proof of demographic fairness.",
            "Paired variants are correlated stress tests, not additional independent examples.",
        ],
        "models": report["models"],
        "settings": report["settings"],
        "seconds": time.perf_counter() - started,
        "comparisons": comparisons,
        "false_supported_ids": support_errors,
        "qualified_needing_review_ids": support_misses,
        "invariance": invariance,
        "paraphrases": paraphrase_checks,
        "slices": slices,
        "evaluation": report,
    }
