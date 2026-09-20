# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch GPT-2 (124M) reimplementation following Karpathy's nanoGPT / "Let's build GPT-2" walkthrough, written as a learning project. Everything lives in a single script, `train.py`; `notebooks/walkthrough.ipynb` is the accompanying notebook. There is no README, test suite, linter, or requirements file.

## Running

Run from the repo root (the data path `datasets/tiny_shakespeare/input.txt` is relative):

```bash
python train.py                                   # single device (auto-detects cuda > mps > cpu)
torchrun --standalone --nproc_per_node=N train.py # DDP, CUDA/NCCL only
```

There are no CLI flags. Hyperparameters (`total_batch_size`, `B`, `T`, `max_lr`, `max_steps`, `warmup_steps`, `use_compile`) are module-level constants in the bottom half of `train.py`; edit them in place.

Development happens on macOS (MPS) but real training runs on a CUDA box (Lambda, `torchrun`). The training loop currently assumes CUDA in places (e.g. an unconditional `torch.cuda.synchronize()` after each step), so it will not run end to end on the Mac without guarding those calls.

## Structure of train.py

The file is a script, not a module: model definitions come first, then top-level training code that executes on import.

- **Model** (top of file): `CausalSelfAttention` → `MLP` → `Block` → `GPT`, configured by the `GPTConfig` dataclass. Attention is the manual masked-softmax version using a registered `bias` buffer (the fused `F.scaled_dot_product_attention` line is commented out). `GPT.from_pretrained` loads HF GPT-2 weights, transposing the Conv1D weights; it filters `.attn.bias` keys, which is why the mask buffer is named `bias`.
- **`GPT.configure_optimizers`**: AdamW with weight decay on 2D+ params only, fused when device type starts with `cuda`.
- **`DataLoaderLite`**: tokenizes the whole tiny-shakespeare file with tiktoken `gpt2` at construction and strides through it, offset by `process_rank` so DDP ranks read disjoint slices.
- **Training script** (bottom): DDP/device setup from `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars, gradient accumulation to a 2**19-token global batch, bf16 autocast, cosine LR with warmup, grad-norm clipping.

## Gotchas

- **`device` is `'cuda:N'` under torchrun but `'cuda'` under plain python.** Any `device == "cuda"` comparison silently fails under DDP; use `.startswith("cuda")`.
- **Model wrapping order matters.** `raw_model` (`model.module` under DDP) must be used for `configure_optimizers`, because the DDP object does not expose `GPT` methods. If `torch.compile` is enabled it should wrap the model *before* DDP, otherwise the `model.require_backward_grad_sync` assignment in the accumulation loop lands on the compile wrapper and DDP syncs gradients on every micro-step.
- **Weight tying:** `wte.weight` and `lm_head.weight` are the same tensor. Init scales residual projections (`NANOGPT_SCALE_INIT` attribute on `c_proj` layers) by `(2 * n_layer) ** -0.5`.
- Training uses `vocab_size=50304` (padded), while `from_pretrained` builds the config with 50257.
- Tensors must be moved to the device by the caller (`x.to(device)`); the model itself takes no device argument and is moved with `model.to(device)`.
- On the Mac, plain `python` may lack torch; the working interpreter is the conda env at `/opt/anaconda3/envs/torchenv/bin/python`.
