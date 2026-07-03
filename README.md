# Kimi-Linear (Gated DeltaNet-2) — a code LLM trained from scratch

A decoder-only language model for **programming code generation**, built from scratch
in JAX / Flax NNX, with a complete, reproducible training pipeline modeled on
[**OpenCoder**](https://arxiv.org/pdf/2411.04905). Everything is here — the model,
the data pipeline, the tokenizer, the training loop, evaluation, and sampling — sized
to run on a single **NVIDIA T4 (16 GB)** for learning, and to scale up to an
**H200** for a serious from-scratch run.

- **Architecture** — the [Kimi Linear](https://arxiv.org/abs/2510.xxxxx) hybrid recipe:
  a 3:1 interleave of cheap **linear-attention** layers and **full-attention** layers.
  The linear mixer is **Gated DeltaNet-2** (decoupled *erase* and *write* gates) in
  place of Kimi's KDA; the full-attention layers are **NoPE Multi-head Latent
  Attention (MLA)**; every block's FFN is a **grouped-GEMM Mixture-of-Experts (MoE)**.
- **Data pipeline** — OpenCoder-style: heuristic filtering → exact + MinHash dedup →
  from-scratch byte-level BPE → Fill-in-the-Middle → packed token corpus → **Grain**
  loader, over **real Hugging Face datasets**.
- **Training** — AdamW + warmup-cosine schedule, gradient accumulation, MoE aux loss +
  aux-loss-free router balancing, checkpointing, optional **bf16 mixed precision**, and
  a two-phase **pretrain → anneal** schedule.

---

## Why this project exists

Most "train an LLM from scratch" repos either (a) wrap a huge framework you can't read,
or (b) train a toy on toy data. This one is meant to be **read end-to-end and actually
run**: every component is a small, annotated JAX/Flax module, the data really comes from
Hugging Face, and the two-phase recipe mirrors how modern code models (OpenCoder,
DeepSeek-Coder, Moonlight/Kimi) are actually trained — just scaled to hardware you can
rent for a few dollars an hour.

---

## How it works

### 1. The architecture (hybrid linear + full attention)

A standard pre-norm decoder stack, `Embed → [DecoderLayer]×N → RMSNorm → LM head`, where
every block is:

```
x = x + TokenMixer(RMSNorm(x))     # attention
x = x + ChannelMixer(RMSNorm(x))   # FFN (always MoE here)
```

The **only** thing that varies across layers is the token mixer, on a fixed **3:1
schedule** (`full_attn_period = 4`):

- **Linear layers (3 of every 4)** — **Gated DeltaNet-2**, an O(L) gated delta-rule
  linear attention. It carries a *fixed-size recurrent state* instead of a growing
  KV-cache, so inference is O(1) per token. It has short causal convs on q/k/v, L2-norm
  on q/k, a channel-wise log-decay `g`, and **decoupled erase (`b`) and write (`w`)
  gates** — the "-2" over vanilla Gated DeltaNet. It has two math forms that are
  verified equivalent in the tests:
  - a **chunkwise** parallel form for training (needs `seq_len % chunk_size == 0`), and
  - a **recurrent** form for streaming decode.
- **Full-attention layers (1 of every 4)** — **NoPE Multi-head Latent Attention (MLA)**
  in absorbed / GQA form, with a low-rank latent KV. Because the linear layers already
  encode position through their recurrence, the MLA layers use **no positional
  encoding**. They keep a growing latent cache for decode.

The **channel mixer on every block is a dispatched grouped-GEMM MoE**: sigmoid routing →
top-k experts → one `jax.lax.ragged_dot` per expert → weighted combine, plus an
always-on shared expert. Load balancing is **aux-loss-free** (a per-expert selection
bias nudged outside the gradient, DeepSeek-V3 style) with a small auxiliary loss on top.

### 2. Two forward modes

- **Training / full sequence** — `model(input_ids)` runs the whole sequence in parallel
  (GDN-2 chunkwise core, MLA full causal matrix) and returns `(logits, aux)` where `aux`
  carries the MoE balancing diagnostics the training loop needs.
- **Streaming / inference** — `model.step(ids, caches)` / `model.generate(...)` reuse
  per-layer state so each new token is O(1) for the linear layers and O(context) for the
  few MLA layers.

### 3. The data pipeline

Real Hugging Face corpora are turned into a packed token stream (OpenCoder Sec. 3):

```
HF dataset ─▶ filter ─▶ dedup ─▶ tokenize ─▶ FIM ─▶ pack ─▶ data/<phase>/{train,val}.bin + meta.json
 (hf_source)  (filters) (exact+   (byte-BPE)  (fim)   (prepare)                         ▲
                         MinHash)                                                        │
                                                                          Grain loader (loader.py)
```

- **Filtering** — OpenCoder-style heuristics (length, alpha ratio, line stats, …).
- **Dedup** — exact (SHA1) **and** near-duplicate (MinHash LSH), no extra services.
- **Tokenizer** — a byte-level BPE trained *from scratch* on your data, with fixed
  special tokens (StarCoder/OpenCoder convention): `<|endoftext|>`, `<|pad|>`,
  `<|fim_prefix|>` / `<|fim_middle|>` / `<|fim_suffix|>`, and repo/file separators.
- **FIM** — a fraction of documents are rewritten into Fill-in-the-Middle order (PSM /
  SPM) so the model learns *infilling*, not just left-to-right completion.
- **Packing** — documents are concatenated into fixed-length `seq_len` windows and
  written as flat `.bin` files with a `meta.json` (vocab size, pad id, tokenizer path).

### 4. The training recipe

`training/train.py` runs **one phase**. The full run is two phases (OpenCoder Sec. 3):

1. **Pretrain** — long run on the large, filtered general corpus.
2. **Anneal** — short run on a smaller, higher-quality corpus, warm-started from the
   final pretrain checkpoint, with a lower peak LR decayed **all the way to zero**.

The loop uses AdamW with a **warmup → cosine** LR schedule, **gradient accumulation**
(via `optax.MultiSteps`, so `max_steps` counts true optimizer steps), the MoE aux loss
added to the cross-entropy, and the aux-loss-free router-bias update applied *after* each
step (outside the gradient). It logs throughput, runs periodic eval (val loss /
perplexity), and checkpoints model + optimizer to `.msgpack`.

### 5. Mixed precision (fp32 ⇄ bf16)

Precision is controlled by one model-config field, `compute_dtype`:

- **`float32`** (default, and the only option on a T4 — Turing has no bf16) — everything
  in fp32.
- **`bfloat16`** (H200 / Ampere+ / TPU) — **master weights, the AdamW state, the loss,
  the GDN-2 recurrent core, all RMSNorms, and the attention softmax stay fp32**; only the
  matmul-heavy **projections and MoE expert GEMMs run in bf16**. This is the standard
  mixed-precision split: ~2× throughput and roughly half the activation memory, with the
  numerically sensitive reductions protected. Params are always stored fp32, so
  checkpoints and `count_params` are unchanged.

---

## What's inside

**Model (the network itself):**

| file | what it is |
|---|---|
| `kimi_linear_gdn2.py` | `KimiLinear` + `KimiLinearConfig`: embed → `[DecoderLayer]×N` → norm → LM head; hybrid 3:1 schedule; streaming `step` / `generate`; `compute_dtype` plumbing. |
| `gated_deltanet_2/core.py` | GDN-2 delta-rule kernel: **chunkwise** (training) + **recurrent** (inference) forms — kept in fp32. |
| `gated_deltanet_2/layer.py` | GDN-2 token mixer: short-conv → SiLU → L2-norm q/k, channel-wise decay `g`, erase gate `b`, write gate `w`, gated-RMSNorm output. |
| `multi_latent_attention/attention.py` | NoPE MLA (absorbed / GQA form) with a growing latent KV cache. |
| `multi_latent_attention/moe.py` | Dispatched grouped-GEMM MoE (`ragged_dot`), shared expert, aux-loss-free bias, `update_router_bias`. |

**Training pipeline (this project):**

```
training/
  config.py            yaml -> KimiLinearConfig + TrainConfig + PrepareConfig
  data/
    hf_source.py       HF dataset -> text documents (text_key or format template)
    filters.py         OpenCoder-style heuristic quality filters
    dedup.py           exact (SHA1) + near-dup (MinHash LSH), no extra deps
    tokenizer.py       from-scratch byte-level BPE + FIM/EOT special tokens
    fim.py             Fill-in-the-Middle (PSM / SPM) transform
    prepare.py         filter -> dedup -> tokenize -> FIM -> pack .bin + meta.json
    train_tokenizer.py train the BPE tokenizer from a data source
    loader.py          Grain DataLoader over the packed corpus
  trainer.py           optimizer, loss (CE + MoE aux), jitted steps, checkpointing
  train.py             run one phase (pretrain / anneal)
  evaluate.py          val loss / perplexity
  generate.py          sampling + FIM-infill decode
configs/               model_{t4,h200} + data_{pretrain,anneal} + {pretrain,anneal}.yaml
scripts/               01..06 thin wrappers for the full run
tests/                 GDN-2 chunk==recurrent + end-to-end pipeline smoke test
Kimi_Linear_GDN2_Colab.ipynb   guided end-to-end run in Colab (with a QUICK_DEMO mode)
```

---

## Configuration

Each training phase is one YAML that references a model YAML:

- **`configs/model_t4.yaml` / `configs/model_h200.yaml`** — map 1:1 onto
  `KimiLinearConfig` (dims, heads, MoE, `max_seq_len`, `compute_dtype`). `vocab_size` is
  overwritten at train time from the tokenizer's `meta.json`.
- **`configs/data_{pretrain,anneal}.yaml`** — the corpus: HF source, filter/dedup/FIM
  settings, output dir.
- **`configs/{pretrain,anneal}.yaml`** — the run: which `model_config`, `seq_len`,
  batch/accum, `max_steps`, LR schedule, logging/eval/ckpt cadence.

Key knobs to reach for first:

| knob | where | effect |
|---|---|---|
| `compute_dtype` | `model_*.yaml` | `float32` (T4) or `bfloat16` (H200) |
| `d_model`, `n_layers`, `moe_n_routed` | `model_*.yaml` | model size (total params scale mostly with `moe_n_routed` and `d_model`) |
| `seq_len` | `{pretrain,anneal}.yaml` | context length — **must be a multiple of `gdn_chunk_size` (64) and ≤ `max_seq_len`** |
| `batch_size` × `grad_accum` | `{pretrain,anneal}.yaml` | tokens per optimizer step & memory |
| `max_steps`, `lr`, `warmup_steps` | `{pretrain,anneal}.yaml` | token budget & schedule |

The model prints `[train] parameters: …` at startup — check it and dial `moe_n_routed` /
`d_model` to fit your memory and time budget.

### Hardware presets

| | **T4 (16 GB)** — `model_t4.yaml` | **H200 (141 GB)** — `model_h200.yaml` |
|---|---|---|
| precision | fp32 | bf16 mixed |
| `d_model` / layers | 512 / 12 | 1024 / 24 |
| GDN-2 heads | 8 × 64 | 16 × 128 |
| MLA q / kv heads | 8 / 2 | 16 / 4 |
| MoE | 8 experts, top-2 | 32 experts, top-8 |
| `seq_len` | 512 | 4096 |
| ~params (total / active) | ~small demo | ~1.9B / ~600M |

> **Token budget.** Roughly Chinchilla-optimal is ~20 tokens/active-param; for code,
> 2–4× beyond that helps. `steps = target_tokens / (batch_size × seq_len × grad_accum)`.
> The default H200 `pretrain.yaml` (`38000` steps × ~524k tokens) is ≈ 20B tokens.

> **MLA memory at long `seq_len`.** The MLA layers materialize the full
> `batch × heads × seq × seq` attention matrix (no flash-attention). At `seq_len=4096`
> that is the largest single tensor in the model; if you OOM, lower `batch_size` first,
> then `seq_len`. bf16 already halves it vs fp32.

---

## Setup

```bash
pip install -r requirements.txt
# then install the JAX build for your hardware:
pip install -U "jax[cuda12]"     # any CUDA 12 GPU (T4, H200, …)
# pip install -U "jax[cpu]"      # CPU (enough for the tests below)
```

## Run it (end to end)

The `scripts/` are thin wrappers around the `python -m training.*` entrypoints.

```bash
# 0) sanity: the parallel training kernel == the recurrent reference; pipeline wires up
JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -m pytest tests -q

# 1) train the tokenizer from the pretraining source (real HF data)
scripts/01_train_tokenizer.sh

# 2) build the packed corpora (downloads from Hugging Face)
scripts/02_prepare_pretrain.sh     # -> data/pretrain/{train,val}.bin + meta.json
scripts/03_prepare_anneal.sh       # -> data/anneal/{train,val}.bin  + meta.json

# 3) Stage-1 pretraining, then Stage-2 annealing (warm-started from pretrain)
scripts/04_pretrain.sh                # -> checkpoints/pretrain
scripts/05_anneal.sh                  # -> checkpoints/anneal  (warm-starts from pretrain)

# 4) generate code (completion or FIM infill)
scripts/06_generate.sh
```

The `scripts/` defaults already match the H200 configs — `05_anneal.sh` warm-starts
from `checkpoints/pretrain/model_38000.msgpack` and `06_generate.sh` samples from
`checkpoints/anneal/model_4000.msgpack` (i.e. `model_<max_steps>.msgpack` for each
phase). If you change `max_steps` or use the T4 preset, pass the right checkpoint
explicitly — `scripts/05_anneal.sh checkpoints/pretrain/model_<STEPS>.msgpack` and
`MODEL=checkpoints/anneal/model_<STEPS>.msgpack scripts/06_generate.sh`. Resume an
interrupted run with `scripts/04_pretrain.sh --resume checkpoints/pretrain`.

Or run everything interactively in **`Kimi_Linear_GDN2_Colab.ipynb`** — it has a
`QUICK_DEMO` switch that shrinks steps/data so the whole pretrain → anneal → generate
loop finishes in minutes (output will be gibberish; it's a wiring test, not a real
model).

## Generation

Completion:

```bash
PYTHONPATH=$PWD python -m training.generate \
  --config configs/anneal.yaml --model checkpoints/anneal/model_4000.msgpack \
  --prompt "def quicksort(arr):" --max-new-tokens 128 --temperature 0.8 --top-p 0.95
```

Fill-in-the-Middle infill (prefix + suffix, model fills the hole):

```bash
PYTHONPATH=$PWD python -m training.generate \
  --config configs/anneal.yaml --model checkpoints/anneal/model_4000.msgpack \
  --prefix "def add(a, b):\n    return " --suffix "\n\nprint(add(2, 3))"
```

> In Colab/Jupyter, inject the checkpoint path with brace-substitution
> (`--model "{MODEL}"`) or an env var, not a bare `$MODEL` — the latter is not
> reliably expanded inside `!` cells.

## Datasets (swappable via YAML)

| stage | default HF dataset | why |
|---|---|---|
| pretrain | `bigcode/the-stack-dedup` (`data/python`, streaming) | large, permissive, real code — scales to a full run |
| anneal | `ise-uiuc/Magicoder-OSS-Instruct-75K` | curated, high-quality problem/solution code |

Point `configs/data_*.yaml` at anything else — a different `the-stack` language dir, or
`codeparrot/github-code-clean` with `streaming: true`. `the-stack-dedup` is **gated**:
accept its terms on Hugging Face and `huggingface-cli login` (or set `HF_TOKEN`) before
`prepare`. Use `bigcode/the-stack-smol` (ungated, tiny) for a quick local trial.

---

## Notes & design deviations (honest)

- **Precision.** The GDN-2 delta-rule core, all RMSNorms/decays, the attention softmax,
  the loss, and the AdamW master weights are always fp32; `compute_dtype: bfloat16` only
  moves the projection + MoE matmuls to bf16. On a T4 (no bf16 tensor cores) keep
  `float32`.
- **Speed.** The GDN-2 core is pure JAX (portable, T4-safe). A fused Triton/Pallas kernel
  would be faster but is not required to train.
- **"Gated DeltaNet-2"** here is the decoupled erase/write variant in `gated_deltanet_2/`
  (separate `b`/`w` gates vs KDA's single `β`). It is a strict generalization of Gated
  DeltaNet; verified equivalent chunk-vs-recurrent in `tests/`.
- **MoE everywhere.** Every block uses the MoE FFN. Total params scale mostly with
  `moe_n_routed`; the T4 preset keeps it tiny (8 experts, top-2).
- **Untied weights** (separate embedding + LM head), as in the DeepSeek/Moonlight lineage
  this model follows. At vocab 32k that is the largest single param block.
```
