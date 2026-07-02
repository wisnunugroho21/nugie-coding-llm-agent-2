"""Training pipeline for the Kimi-Linear (GDN-2) code language model.

Package layout
--------------
  training.config      : yaml-backed config objects; builds a KimiLinearConfig.
  training.data.*      : OpenCoder-style data pipeline
                         (tokenizer -> filter -> dedup -> FIM -> pack -> Grain loader).
  training.trainer     : optimizer, loss (CE + MoE aux), jitted train/eval steps, ckpt.
  training.train       : entry point running a pretrain or anneal phase.
  training.evaluate    : validation loss / perplexity.
  training.generate    : sampling + FIM infill decode.

Everything targets a single NVIDIA T4 (16 GB). See README.md for the command order.
"""
