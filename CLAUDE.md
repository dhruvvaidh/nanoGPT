# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch GPT-2 (124M) reimplementation following Karpathy's nanoGPT / "Let's build GPT-2" walkthrough, written as a learning project. `train.py` is the entry point; the model, data loading, evals and DDP setup live in sibling modules. `notebooks/walkthrough.ipynb` pokes at HF GPT-2 weights (position embeddings, attention matrices) and is exploratory only. The two files under `datasets/` are dataset prep/eval helpers. There is no README, test suite, linter, or requirements file.

**The long explanatory comments are the point of this repo** — they are the author's notes on why each optimization exists. Never delete or paraphrase them when refactoring; move them with the code they describe. Same for the commented-out alternatives (manual masked-softmax attention, the tiny-shakespeare loader, the hand-written generation block at the bottom of `train.py`).

A full 19,073-step run has already been done (final val loss ~3.40, HellaSwag acc_norm ~0.27); `log/log.txt` and the `log/model_*.pt` checkpoints from it are on disk but gitignored.

## Running

Run from the repo root — the shard path `datasets/fineWeb/edu_fineweb10B` is relative to the cwd, not to `train.py`.

```bash
python train.py                                   # single device (auto-detects cuda > mps > cpu)
torchrun --standalone --nproc_per_node=N train.py # DDP, CUDA/NCCL only
```

There are no CLI flags. Hyperparameters (`total_batch_size`, `B`, `T`, `max_lr`, `max_steps`, `warmup_steps`, `use_compile`, `eval_interval`, `checkpoint_interval`) are module-level constants in the **top** of `train.py`, above `main()`; edit them in place. `get_lr` reads them as globals, so overriding them from another module before calling `main()` works.

Data prep, required before `train.py` will start:

```bash
cd datasets/fineWeb && python fineweb.py          # ~19GB of .npy shards into edu_fineweb10B/ (gitignored)
```

`fineweb.py` writes 100M-token uint16 shards named `edufineweb_{train,val}_NNNNNN.npy`; shard 0 is the only `val` shard. `DataLoaderLite` selects shards by substring match on the split name.

HellaSwag needs no prep — `iterate_examples` downloads the val set on first use into `datasets/hellaSwag/hellaswag/`. To get a reference number from real HF GPT-2 weights:

```bash
python datasets/hellaSwag/hellaswag.py -m gpt2
```

Development happens on macOS (MPS) but real training runs on a CUDA box (Lambda, `torchrun`). The training loop still assumes CUDA in places (an unconditional `torch.cuda.synchronize()` per step), so it will not run end to end on the Mac without guarding those calls.

## Module layout

Five flat modules at the repo root. Nothing runs at import time — `train.py` guards the loop behind `if __name__ == "__main__": main()`, so any module can be imported safely (useful for poking at pieces from a REPL or the notebook).

- **`model.py`** — `CausalSelfAttention` → `MLP` → `Block` → `GPT`, configured by the `GPTConfig` dataclass.
  - Attention uses `F.scaled_dot_product_attention(..., is_causal=True)` (flash); the manual masked-softmax version and its `bias` mask buffer are both commented out.
  - `GPT.from_pretrained` loads HF GPT-2 weights, transposing the Conv1D weights.
  - `GPT.configure_optimizers(weight_decay, learning_rate, device_type, master_process=True)` — AdamW with weight decay on 2D+ params only, fused when `device_type` starts with `cuda`.
- **`dataloader.py`** — `load_tokens` plus `DataLoaderLite`, which streams `.npy` shards strided by `process_rank` so DDP ranks read disjoint slices, wrapping to the next shard when a batch would overrun. Constructed once per split (`train`, `val`).
- **`evaluation.py`** — the three "once in a while" phases, each returning its numbers so the caller decides what gets printed/logged: `estimate_val_loss`, `evaluate_hellaswag`, `generate_samples`. Also `get_most_likely_row` (HellaSwag scoring — per-row mean completion loss, argmin picks the predicted ending, the `acc_norm` metric) and the shared `enc` tokenizer. This module owns the `sys.path` hack and the `hellaswag` import.
- **`distributed.py`** — `setup_distributed()` reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, initializes NCCL, autodetects the device, and returns a `DistributedContext` dataclass (`ddp`, `ddp_rank`, `ddp_local_rank`, `ddp_world_size`, `device`, `device_type`, `master_process`). `cleanup_distributed(ddp)` tears it down.
- **`train.py`** — hyperparameters, `get_lr`, and `main()`: the gradient-accumulation loop to a 2**19-token global batch, bf16 autocast, cosine LR with warmup, grad-norm clipping, logging and checkpointing.

`master_process` is threaded explicitly into `configure_optimizers` and `DataLoaderLite` (both default to `True`) instead of being read off a module global — that global is what previously pinned them to the training script's scope.

Every `eval_interval` (250) steps and on the last step, `main()` runs three evals in order: val loss over 20 batches, HellaSwag over the val set sharded across ranks, and a 4-sample generation from `"Hello, I'm a language model,"`. Checkpoints are written every `checkpoint_interval` (5000) steps to `log/model_NNNNN.pt`; per-step metrics append to `log/log.txt` as `{step} {train|val|hella} {value}`.

`model.eval()` is called inside `estimate_val_loss` and `generate_samples` but deliberately not inside `evaluate_hellaswag` — it relies on the val eval having just run on the same step, which is how the original inline code behaved. `main()` calls `model.train()` before the accumulation loop.

## Gotchas

- **`device` is `'cuda:N'` under torchrun but `'cuda'` under plain python.** Any `device == "cuda"` comparison silently fails under DDP; use `.startswith("cuda")` or the derived `device_type`, which is what `torch.autocast` needs.
- **`require_backward_grad_sync` must be set before the forward pass**, not just before `backward()` — DDP reads it inside `forward()` to decide whether to arm the backward hooks. Setting it after the forward means gradients sync on every micro-step.
- **Model wrapping order matters.** `torch.compile` must wrap the model *before* DDP, for the same reason: otherwise `model.require_backward_grad_sync` lands on the compile wrapper. `raw_model` (`model.module` under DDP) must be used for `configure_optimizers` and checkpointing, because the DDP object does not expose `GPT` methods.
- **`use_compile` gates the evals.** HellaSwag and sampling are skipped when `use_compile` is True (recompiles on the varying shapes). It is currently `False`.
- **Importing hellaswag needs a `sys.path` hack.** The HuggingFace `datasets` package shadows the local `datasets/` folder, so `evaluation.py` appends `datasets/hellaSwag` to `sys.path` and imports `hellaswag` as a top-level module. Don't "clean this up" into `from datasets.hellaSwag import ...`.
- **`from_pretrained` still filters `.attn.bias` keys** even though the mask buffer is no longer registered; that filter is why the buffer was named `bias` in the first place.
- **Weight tying:** `wte.weight` and `lm_head.weight` are the same tensor. Init scales residual projections (`NANOGPT_SCALE_INIT` attribute on `c_proj` layers) by `(2 * n_layer) ** -0.5`, and zeroes all Linear biases.
- Training uses `vocab_size=50304` (padded), while `from_pretrained` builds the config with 50257.
- **No resume support.** `log/log.txt` is truncated at startup and checkpoints save only model/config/step/val_loss — no optimizer state or RNG, so a run cannot be continued as-is.
- Tensors must be moved to the device by the caller (`x.to(device)`); the model itself takes no device argument and is moved with `model.to(device)`.
- `log/` and `datasets/fineWeb/edu_fineweb10B/` are gitignored. `.gitignore` also lists `CLAUDE.md`, but this file is already tracked so the entry has no effect.
- On the Mac, plain `python` may lack torch; the working interpreter is the conda env at `/opt/anaconda3/envs/torchenv/bin/python`.
