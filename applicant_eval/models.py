"""Local model adapters; no applicant data is sent to a hosted inference service."""

import re

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


def unit(v):
    v = np.asarray(v, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError("model returned non-finite embeddings")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


class LocalModels:
    def __init__(self, device="cpu", offline=False):
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        # Bounded CPU parallelism avoids severe oversubscription on laptops/CI.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.encoder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            local_files_only=offline,
            trust_remote_code=False,
            revision=EMBEDDING_REVISION,
        )
        self.verifier = CrossEncoder(
            NLI_MODEL,
            device=device,
            local_files_only=offline,
            trust_remote_code=False,
            revision=NLI_REVISION,
        )
        labels = {str(v).lower(): int(k) for k, v in self.verifier.model.config.id2label.items()}
        if set(labels) != {"contradiction", "entailment", "neutral"}:
            raise ValueError(f"unexpected NLI label mapping: {labels}")
        self.label_order = [labels[x] for x in ("contradiction", "entailment", "neutral")]
        self._vectors = {}
        self._checks = {}
        self.metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "evidence_model": NLI_MODEL,
            "evidence_revision": NLI_REVISION,
            "device": device,
        }

    def spans(self, text):
        """Overlapping, token-bounded windows with exact source offsets, including the tail."""
        size = min(160, self.encoder.max_seq_length - 2)
        stride = size - 32
        if stride <= 0:
            raise ValueError("embedding context is too short")
        # Keep short claims separate so a relevant final sentence is not diluted
        # by a long unrelated work history. Oversize sentences still use windows.
        boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", text)] + [len(text)]
        for left, right in zip(boundaries, boundaries[1:]):
            tokens = self.encoder.tokenizer(
                text[left:right],
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
                verbose=False,
            )
            offsets = tokens["offset_mapping"]
            for start in range(0, len(offsets), stride):
                end = min(start + size, len(offsets))
                a, b = left + offsets[start][0], left + offsets[end - 1][1]
                if b > a:
                    yield a, b
                if end == len(offsets):
                    break

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self._vectors))
        if missing:
            vectors = self.encoder.encode(
                missing,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._vectors.update(zip(missing, unit(vectors)))
        return np.array([self._vectors[t] for t in texts])

    def verify(self, pairs):
        """Return [contradiction, entailment, neutral]; these are uncalibrated model scores."""
        import torch

        missing = list(dict.fromkeys(pair for pair in pairs if pair not in self._checks))
        if missing:
            # Fail explicitly instead of silently dropping evidence/hypotheses at truncation.
            limit = min(self.verifier.tokenizer.model_max_length, 512)
            for premise, hypothesis in missing:
                if (
                    len(self.verifier.tokenizer(premise, hypothesis, truncation=False)["input_ids"])
                    > limit
                ):
                    raise ValueError(
                        "evidence pair exceeds the verifier context; shorten evidence windows"
                    )
            values = self.verifier.predict(
                missing,
                batch_size=32,
                show_progress_bar=False,
                activation_fn=torch.nn.Softmax(dim=1),
                convert_to_numpy=True,
            )
            values = np.asarray(values)[:, self.label_order]
            if not np.isfinite(values).all():
                raise ValueError("verifier returned non-finite scores")
            self._checks.update(zip(missing, values))
        return np.array([self._checks[pair] for pair in pairs])
