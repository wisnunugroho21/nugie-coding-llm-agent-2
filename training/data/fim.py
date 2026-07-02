"""Fill-in-the-Middle (FIM) transform.

Code models are trained with a mix of left-to-right and infilling objectives so
that, at inference, they can complete code given both a prefix AND a suffix (the
common IDE "type in the middle of a function" case). OpenCoder and the StarCoder
lineage apply FIM to a fraction of documents by splitting each into
(prefix, middle, suffix) and reordering with sentinel tokens.

We apply FIM at the *token* level (operate on an already-encoded id list) in the
two canonical orderings:

  PSM:  <prefix> P <suffix> S <middle> M middle
  SPM:  <prefix> <suffix> S prefix <middle> M middle      (suffix-prefix-middle)

so the model learns to emit the middle span conditioned on both sides. At inference
you build the prompt up to (and including) the `<|fim_middle|>` token and decode.
"""

from __future__ import annotations

import numpy as np


def maybe_fim(
    ids: list[int],
    fim_prefix_id: int,
    fim_middle_id: int,
    fim_suffix_id: int,
    rng: np.random.Generator,
    rate: float = 0.5,
    spm_rate: float = 0.5,
) -> list[int]:
    """Return `ids` unchanged with prob (1-rate), else a FIM-reordered version.

    Splits at two random cut points into prefix/middle/suffix, then interleaves the
    three FIM sentinels. Documents shorter than 3 tokens are left as-is.
    """
    if rng.random() >= rate or len(ids) < 3:
        return ids

    n = len(ids)
    a, b = sorted(int(x) for x in rng.integers(0, n + 1, size=2))
    prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]

    if rng.random() < spm_rate:
        # SPM: suffix first, then prefix, then middle.
        return (
            [fim_prefix_id, fim_suffix_id]
            + suffix
            + [fim_middle_id]
            + prefix
            + middle
        )
    # PSM: prefix, suffix, middle.
    return (
        [fim_prefix_id]
        + prefix
        + [fim_suffix_id]
        + suffix
        + [fim_middle_id]
        + middle
    )
