# Kimi-Linear (Gated DeltaNet-2) — a code LLM trained from scratch

A decoder-only language model for **programming code generation**, built from scratch
in JAX / Flax NNX, with a complete training pipeline modeled on
[**OpenCoder**](https://arxiv.org/pdf/2411.04905) and sized to train on a single
**NVIDIA T4 (16 GB)**.

- **Architecture** — the [Kimi Linear](https://arxiv.org/abs/2510.xxxxx) hybrid recipe:
  a 3:1 interleave of cheap **linear-attention** layers and **full-attention** layers.
  The linear mixer is **Gated DeltaNet-2** (decoupled *erase* and *write* gates) in
  place of Kimi's KDA; the full-attention layers are **NoPE Multi-head Latent
  Attention (MLA)**; every block's FFN is a **grouped-GEMM MoE**.
- **Data pipeline** — OpenCoder-style: heuristic filtering → exact + MinHash dedup →
  from-scratch byte-level BPE → Fill-in-the-Middle → packed token corpus → **Grain**
  loader, over **real Hugging Face datasets**.
- **Training** — AdamW + warmup-cosine, gradient accumulation, MoE aux loss +
  aux-loss-free router balancing, checkpointing, and a two-phase
  **pretrain → anneal** schedule.

---

## Model (already implemented)

| file | what it is |
|---|---|
| `kimi_linear_gdn2.py` | `KimiLinear`: embed → `[DecoderLayer]×N` → norm → LM head; hybrid schedule + streaming `generate`. |
| `gated_deltanet_2/core.py` | GDN-2 delta-rule kernel: **chunkwise** (training) + **recurrent** (inference) forms. |
| `gated_deltanet_2/layer.py` | GDN-2 token mixer: short-conv → SiLU → L2-norm q/k, channel-wise decay `g`, erase gate `b`, write gate `w`, gated-RMSNorm output. |
| `multi_latent_attention/attention.py` | NoPE MLA (absorbed / GQA form) with a growing latent KV cache. |
| `multi_latent_attention/moe.py` | Dispatched grouped-GEMM MoE (`ragged_dot`), shared expert, aux-loss-free bias. |

Default T4 architecture (`configs/model_t4.yaml`): `d_model=512`, `12` layers
(indices 3/7/11 are MLA), GDN-2 `8×64` heads, MLA `8` q / `2` kv heads, MoE `8`
experts top-2, `seq_len=512`.

## Training pipeline (this project)

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
configs/               model_t4 + data_{pretrain,anneal} + {pretrain,anneal}.yaml
scripts/               01..06 thin wrappers for the full run
tests/                 GDN-2 chunk==recurrent + end-to-end pipeline smoke test
```

---

## Setup

```bash
pip install -r requirements.txt
# then install the JAX build for your hardware:
pip install -U "jax[cuda12]"     # NVIDIA T4 / any CUDA 12 GPU
# pip install -U "jax[cpu]"      # CPU (for the tests below)
```

## Run it (end to end)

```bash
# 0) sanity: the parallel training kernel == the recurrent reference; pipeline wires up
JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -m pytest tests -q

# 1) train the tokenizer from the pretraining source (real HF data)
scripts/01_train_tokenizer.sh

# 2) build the packed corpora (downloads from Hugging Face)
scripts/02_prepare_pretrain.sh     # bigcode/the-stack-smol  -> data/pretrain/*.bin
scripts/03_prepare_anneal.sh       # Magicoder OSS-Instruct   -> data/anneal/*.bin

# 3) Stage-1 pretraining, then Stage-2 annealing (warm-started from pretrain)
scripts/04_pretrain.sh                                     # -> checkpoints/pretrain
scripts/05_anneal.sh checkpoints/pretrain/model_20000.msgpack   # -> checkpoints/anneal

# 4) generate code (completion or FIM infill)
MODEL=checkpoints/anneal/model_3000.msgpack scripts/06_generate.sh
```

FIM infill directly:

```bash
PYTHONPATH=$PWD python -m training.generate \
  --config configs/anneal.yaml --model checkpoints/anneal/model_3000.msgpack \
  --prefix "def add(a, b):\n    return " --suffix "\n\nprint(add(2, 3))"
```

## Datasets (swappable via YAML)

| stage | default HF dataset | why |
|---|---|---|
| pretrain | `bigcode/the-stack-smol` (`data/python`) | small, permissive, real code; laptop/T4-sized |
| anneal | `ise-uiuc/Magicoder-OSS-Instruct-75K` | curated, high-quality problem/solution code |

Point `configs/data_*.yaml` at anything else — e.g. more `the-stack-smol` language
dirs, or `codeparrot/github-code-clean` with `streaming: true` — to scale the corpus.

---

## Notes for the T4

- **Precision.** The T4 (Turing) has fp16 tensor cores but **no bf16**. This pipeline
  runs in **fp32** by default — the GDN-2 delta-rule core needs fp32 internally, and
  fp32 avoids loss-scaling bookkeeping. For more speed, cast the *compute* to fp16 and
  add `optax` dynamic loss scaling; keep the GDN-2 core and all norms/decays in fp32.
- **Memory.** Tune `batch_size` × `grad_accum` in `configs/pretrain.yaml`. The MLA
  layers are O(L²); at `seq_len=512` with the default MoE this fits comfortably in
  16 GB. Raise `seq_len` (keep it a multiple of `gdn_chunk_size=64` and ≤ `max_seq_len`)
  or shrink `moe_n_routed` to trade memory.
- **Speed.** The GDN-2 core here is pure JAX (portable, T4-safe). A fused Triton/Pallas
  kernel would be faster but is not required to train.

## Design deviations (honest notes)

- **"Gated DeltaNet-2"** here is the decoupled erase/write variant already implemented
  in `gated_deltanet_2/` (separate `b`/`w` gates vs KDA's single `β`). It is a strict
  generalization of Gated DeltaNet; verified equivalent chunk-vs-recurrent in tests.
- **MoE everywhere.** Following the existing model, every block uses the MoE FFN. The
  T4 preset keeps it tiny (8 experts, top-2) so total params stay modest.
- **Weights are untied** (separate embedding + LM head), as in the DeepSeek/Moonlight
  lineage this model follows. At vocab 32k that is the largest single param block.
```
