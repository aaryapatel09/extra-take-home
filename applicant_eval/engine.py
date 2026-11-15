"""Multiple profile centroids, source-linked evidence checks, and a review queue."""

import hashlib
import math
from dataclasses import asdict

import numpy as np

from .context import context_flags
from .models import unit
from .schema import InputError, Settings, unique


def chunks(applicant, models):
    # Repeating a claim must not crowd other evidence out of the retrieval budget.
    seen = set()
    result = []
    for e in sorted(applicant.evidence, key=lambda e: e.id):
        for start, end in models.spans(e.text):
            quote = e.text[start:end]
            if quote not in seen:
                result.append({"evidence_id": e.id, "start": start, "end": end, "quote": quote})
                seen.add(quote)
    return result


def document_vector(text, models):
    spans = [text[a:b] for a, b in models.spans(text)]
    if not spans:
        raise InputError("text has no embeddable content")
    return unit(models.encode(spans).mean(axis=0))


def evaluate(role, applicants, models, settings=None):
    settings = settings or Settings()
    unique([a.id for a in applicants], "applicant IDs")
    prototypes = {}
    negatives = {}
    all_examples = []
    for profile in role.profiles:
        examples = [document_vector(t, models) for t in profile.examples]
        all_examples.extend(examples)
        prototypes[profile.id] = unit(np.mean(examples, axis=0))
        negatives[profile.id] = [document_vector(t, models) for t in profile.near_misses]
    single = unit(np.mean(all_examples, axis=0))
    negative_examples = [v for values in negatives.values() for v in values]
    single_negative = (
        unit(np.mean(negative_examples, axis=0)) if negative_examples else np.zeros_like(single)
    )
    job = document_vector(role.description, models)
    queries = models.encode([c.hypothesis for c in role.criteria])
    prepared = []
    all_pairs = []
    for applicant in applicants:
        source_chunks = chunks(applicant, models)
        vectors = models.encode([c["quote"] for c in source_chunks]) if source_chunks else None
        candidate = unit(vectors.mean(axis=0)) if source_chunks else np.zeros_like(single)
        retrieved = []
        for query, criterion in zip(queries, role.criteria):
            similarities = vectors @ query if source_chunks else np.array([])
            indices = sorted(range(len(similarities)), key=lambda i: (-float(similarities[i]), i))[
                : settings.evidence_top_k
            ]
            references = []
            for i in indices:
                pair = (source_chunks[i]["quote"], criterion.hypothesis)
                references.append((len(all_pairs), source_chunks[i], float(similarities[i])))
                all_pairs.append(pair)
            retrieved.append(references)
        prepared.append((applicant, candidate, vectors, retrieved))
    verified = models.verify(all_pairs) if all_pairs else np.empty((0, 3))
    if verified.shape != (len(all_pairs), 3) or not np.isfinite(verified).all():
        raise ValueError("invalid evidence model output")
    if (
        np.any(verified < 0)
        or np.any(verified > 1)
        or (len(verified) and not np.allclose(verified.sum(axis=1), 1, atol=1e-4))
    ):
        raise ValueError("evidence model output is not a probability distribution")
    results = []
    for applicant, candidate, vectors, retrieved in prepared:
        evidence = []
        for criterion, references in zip(role.criteria, retrieved):
            matches = []
            for i, source, similarity in references:
                contradiction, support, neutral = map(float, verified[i])
                matches.append(
                    {
                        **source,
                        "retrieval_similarity": similarity,
                        "support_score": support,
                        "contradiction_score": contradiction,
                        "neutral_score": neutral,
                        "context_flags": context_flags(source["quote"]),
                    }
                )
            relevant = [
                m for m in matches if m["retrieval_similarity"] >= settings.min_evidence_similarity
            ]
            supports = [
                m
                for m in relevant
                if m["support_score"] >= settings.evidence_threshold and not m["context_flags"]
            ]
            # NLI contradiction alone is not evidence of lacking a capability: a
            # non-exhaustive resume statement may simply describe unrelated work.
            conflicts = [
                m
                for m in relevant
                if m["contradiction_score"] >= settings.contradiction_threshold
                and m["context_flags"] == ["explicit_negation"]
            ]
            status = (
                "conflicting"
                if supports and conflicts
                else "contradicted"
                if conflicts
                else "supported"
                if supports
                else "unknown"
            )
            evidence.append(
                {
                    "criterion_id": criterion.id,
                    "description": criterion.description,
                    "required": criterion.required,
                    "status": status,
                    "sources": matches,
                }
            )
        by_id = {e["criterion_id"]: e for e in evidence}
        profile_results = []
        for p in role.profiles:
            similarity = float(candidate @ prototypes[p.id])
            chunk_similarity = (
                float(np.max(vectors @ prototypes[p.id])) if vectors is not None else 0.0
            )
            near_similarity = max((float(candidate @ n) for n in negatives[p.id]), default=None)
            supported = sum(by_id[c]["status"] == "supported" for c in p.criteria)
            profile_results.append(
                {
                    "profile_id": p.id,
                    "description": p.description,
                    "similarity": similarity,
                    "best_chunk_similarity": chunk_similarity,
                    "near_miss_similarity": near_similarity,
                    "contrastive_margin": similarity - near_similarity
                    if near_similarity is not None
                    else None,
                    "criteria": list(p.criteria),
                    "supported_fraction": supported / len(p.criteria),
                }
            )
        closest = max(profile_results, key=lambda p: (p["similarity"], p["profile_id"]))
        best = max(
            profile_results,
            key=lambda p: (p["supported_fraction"], p["similarity"], p["profile_id"]),
        )
        core = [e for e in evidence if e["required"]]
        coverage = sum(e["status"] == "supported" for e in core) / len(core)
        supported = coverage == 1 and best["supported_fraction"] == 1
        results.append(
            {
                "applicant_id": applicant.id,
                "review_status": "evidence_supported" if supported else "needs_review",
                "core_supported_fraction": coverage,
                "best_supported_profile": best["profile_id"],
                "best_profile_supported_fraction": best["supported_fraction"],
                "closest_profile": closest["profile_id"],
                "profile_similarity": closest["similarity"],
                "baselines": {
                    "job_description": float(candidate @ job),
                    "single_centroid": float(candidate @ single),
                    "contrastive_single_centroid": float(
                        candidate @ single - candidate @ single_negative
                    ),
                    "multiple_centroids": closest["similarity"],
                    "contrastive_multiple_centroids": max(
                        p["contrastive_margin"]
                        if p["contrastive_margin"] is not None
                        else p["similarity"]
                        for p in profile_results
                    ),
                    "chunk_matching": max(p["best_chunk_similarity"] for p in profile_results),
                },
                "profiles": profile_results,
                "evidence": evidence,
                "unresolved_criteria": [
                    e["criterion_id"]
                    for e in evidence
                    if e["status"] != "supported"
                    and (e["required"] or e["criterion_id"] in best["criteria"])
                ],
                "limitations": [
                    "Evidence consists of applicant claims, not independently verified facts.",
                    "Similarity and NLI scores are not probabilities of job performance.",
                ],
            }
        )
    results.sort(
        key=lambda r: (
            -(r["review_status"] == "evidence_supported"),
            -r["core_supported_fraction"],
            -r["best_profile_supported_fraction"],
            -r["profile_similarity"],
            r["applicant_id"],
        )
    )
    for rank, result in enumerate(results, 1):
        result["rank"] = rank
    return {
        "schema_version": 1,
        "role_id": role.id,
        "role_title": role.title,
        "purpose": "Human review support; no automatic advancement or rejection.",
        "models": models.metadata,
        "settings": asdict(settings),
        "results": results,
    }


def review_queue(results, budget, exploration_fraction=0.2, seed=42):
    if type(budget) is not int or budget < 0:
        raise InputError("review budget must be a nonnegative integer")
    if not math.isfinite(exploration_fraction) or not 0 <= exploration_fraction <= 1:
        raise InputError("exploration_fraction must be in [0, 1]")
    count = min(budget, len(results))
    reserve = min(count, math.ceil(count * exploration_fraction)) if count > 1 else 0
    primary = results[: count - reserve]
    # Hash sampling is stable across input ordering and Python versions.
    remaining = sorted(
        results[count - reserve :],
        key=lambda r: hashlib.sha256(f"{seed}:{r['applicant_id']}".encode()).hexdigest(),
    )
    return [{"applicant_id": r["applicant_id"], "reason": "ranked_review"} for r in primary] + [
        {"applicant_id": r["applicant_id"], "reason": "exploration_review"}
        for r in remaining[:reserve]
    ]
