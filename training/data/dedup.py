"""Deduplication: exact (hash) + near-duplicate (MinHash LSH).

OpenCoder (Sec. 2.1) reports that deduplication — especially *file-level* fuzzy
dedup — is the single most impactful cleaning step for code corpora. We implement
both stages with no third-party dependency:

  * ExactDeduper : SHA-1 over whitespace-normalized text; drops byte-identical files.
  * MinHashLSH   : shingled MinHash + banded LSH; drops near-duplicates above a
                   Jaccard threshold (documents that differ only in comments,
                   headers, or trivial edits).

Both are in-memory and single-process — appropriate for the T4-scale corpora this
pipeline targets (tens to low-hundreds of thousands of files). For web-scale data
you would shard this across processes; the interface (`add_if_new(text) -> bool`)
would stay the same.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_WS = re.compile(r"\s+")
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


class ExactDeduper:
    """Drops byte-identical (after whitespace normalization) documents."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def add_if_new(self, text: str) -> bool:
        h = hashlib.sha1(_normalize(text).encode("utf-8", "ignore")).hexdigest()
        if h in self._seen:
            return False
        self._seen.add(h)
        return True


class MinHashLSH:
    """Near-duplicate detector via MinHash signatures + banded LSH.

    num_perm permutations -> signature of length num_perm. Split into `bands` bands
    of `rows = num_perm // bands` rows; two documents that agree on *any* whole band
    are candidate duplicates. `bands` is chosen from the target Jaccard threshold via
    the standard LSH approximation threshold ~= (1/bands) ** (1/rows).
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 64,
        ngram: int = 5,
        seed: int = 1,
    ) -> None:
        self.num_perm = num_perm
        self.ngram = ngram
        self.bands = self._pick_bands(threshold, num_perm)
        self.rows = num_perm // self.bands
        rng = np.random.default_rng(seed)
        # Random affine hashes h_i(x) = (a_i * x + b_i) mod p, used for permutations.
        self._a = rng.integers(1, _MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, _MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self._buckets: list[dict[bytes, int]] = [dict() for _ in range(self.bands)]

    @staticmethod
    def _pick_bands(threshold: float, num_perm: int) -> int:
        """Largest #bands dividing num_perm whose LSH threshold <= target."""
        best = 1
        for b in range(1, num_perm + 1):
            if num_perm % b:
                continue
            rows = num_perm // b
            t = (1.0 / b) ** (1.0 / rows)
            if t <= threshold:
                best = b
                break
            best = b
        return best

    def _shingles(self, text: str) -> np.ndarray:
        toks = _normalize(text).split(" ")
        if len(toks) < self.ngram:
            grams = [" ".join(toks)] if toks else [""]
        else:
            grams = [" ".join(toks[i : i + self.ngram]) for i in range(len(toks) - self.ngram + 1)]
        # 32-bit hash of each shingle.
        hs = {int.from_bytes(hashlib.sha1(g.encode("utf-8", "ignore")).digest()[:4], "little") for g in grams}
        return np.fromiter(hs, dtype=np.uint64, count=len(hs))

    def _signature(self, text: str) -> np.ndarray:
        sh = self._shingles(text)  # [S]
        if sh.size == 0:
            sh = np.zeros(1, dtype=np.uint64)
        # (a[:,None]*sh + b) mod p, then min over shingles -> [num_perm].
        hashed = (self._a[:, None] * sh[None, :] + self._b[:, None]) % _MERSENNE_PRIME
        return (hashed & _MAX_HASH).min(axis=1).astype(np.uint64)

    def add_if_new(self, text: str) -> bool:
        """True (and registers it) if no near-duplicate has been seen; False if dup."""
        sig = self._signature(text)
        # Each band has its own bucket dict, so the band payload alone is the key.
        band_keys = [
            sig[bi * self.rows : (bi + 1) * self.rows].tobytes()
            for bi in range(self.bands)
        ]
        if any(band_keys[bi] in self._buckets[bi] for bi in range(self.bands)):
            return False
        for bi, band in enumerate(band_keys):
            self._buckets[bi][band] = 1
        return True


class Deduper:
    """Combined exact + optional near-dedup with a single `add_if_new` gate."""

    def __init__(
        self,
        exact: bool = True,
        minhash: bool = True,
        threshold: float = 0.8,
        num_perm: int = 64,
        ngram: int = 5,
        seed: int = 1,
    ) -> None:
        self.exact = ExactDeduper() if exact else None
        self.minhash = (
            MinHashLSH(threshold, num_perm, ngram, seed) if minhash else None
        )

    def add_if_new(self, text: str) -> bool:
        if self.exact is not None and not self.exact.add_if_new(text):
            return False
        if self.minhash is not None and not self.minhash.add_if_new(text):
            return False
        return True
