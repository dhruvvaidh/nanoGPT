# Architecture reference

This describes how the code works **as it is now**. For how it got this way, see [`BUILD_LOG.md`](BUILD_LOG.md). For how to run it, see the [README](../README.md).

## Module map

```
train.py ──┬── distributed.py   setup_distributed() → DistributedContext, cleanup_distributed()
           ├── model.py         GPTConfig, GPT (+ Block, CausalSelfAttention, MLP)
           ├── dataloader.py    DataLoaderLite (reads datasets/fineWeb/edu_fineweb10B/*.npy)
           └── evaluation.py ── datasets/hellaSwag/hellaswag.py  (via sys.path, see Gotchas)
```

Nothing runs at import time. `train.py`'s loop is behind `if __name__ == "__main__": main()`, so any module can be imported from a REPL or the notebook.

## 1. The model (`model.py`)

### Data flow

```
idx (B, T) token ids
  │
  ├─ wte(idx)            (B, T, 768)   token embedding       ┐ summed
  ├─ wpe(arange(T))      (T, 768)      learned position emb  ┘ (broadcast over B)
  ▼
x (B, T, 768)  ──►  Block × 12  ──►  ln_f  ──►  lm_head  ──►  logits (B, T, vocab)
                                                   ▲
                                     shares its weight with wte (weight tying)

loss = cross_entropy(logits.view(B*T, vocab), targets.view(B*T))   # if targets given
returns (logits, loss)
```

### A Block (pre-LayerNorm residual)

```
x ──┬──────────────────────────────►(+)──┬──────────────────────────────►(+)──► out
    └─► ln_1 ─► CausalSelfAttention ─┘   └─► ln_2 ─► MLP ───────────────┘
        "reduce": tokens talk               "map": each token on its own
```

GPT-2 puts LayerNorm at the **input** of each sub-layer (pre-LN, unlike the original Transformer's post-LN) and adds a final `ln_f`. This keeps a clean residual path from the output all the way back to the embeddings, which makes the network easier to optimize.

### CausalSelfAttention shapes (GPT-2 small: C=768, n_head=12, head_size=64)

| Step | Shape |
|---|---|
| input `x` | `(B, T, 768)` |
| `qkv = c_attn(x)` | `(B, T, 2304)` |
| `q, k, v = qkv.split(768, dim=2)` | 3 × `(B, T, 768)` |
| `.view(B, T, 12, 64).transpose(1, 2)` | 3 × `(B, 12, T, 64)`, so heads act like a batch dimension |
| `F.scaled_dot_product_attention(q, k, v, is_causal=True)` | `(B, 12, T, 64)` |
| `.transpose(1, 2).contiguous().view(B, T, 768)` | `(B, T, 768)`, heads concatenated |
| `c_proj` | `(B, T, 768)` |

`is_causal=True` means token *t* only attends to tokens ≤ *t*. The SDPA call runs FlashAttention on supported GPUs. The equivalent naive version (explicit `T×T` matrix, `masked_fill` with a `tril` mask, softmax, `@ v`) is kept as comments.

### MLP

`c_fc: 768 → 3072` → `GELU(approximate='tanh')` → `c_proj: 3072 → 768`. The hidden layer is 4× wider, as in the original Transformer.

### Configuration

```python
@dataclass
class GPTConfig:
    block_size: int = 1024   # max sequence length (size of wpe)
    vocab_size: int = 50257  # training overrides this to 50304
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768
```

| Size | `n_layer` | `n_head` | `n_embd` | Params |
|---|---|---|---|---|
| `gpt2` (trained here) | 12 | 12 | 768 | 124M |
| `gpt2-medium` | 24 | 16 | 1024 | 350M |
| `gpt2-large` | 36 | 20 | 1280 | 774M |
| `gpt2-xl` | 48 | 25 | 1600 | 1558M |

### Parameter count (training config, `vocab_size=50304`)

| Component | Parameters |
|---|---|
| `wte` 50304 × 768 (shared with `lm_head`) | 38,633,472 |
| `wpe` 1024 × 768 | 786,432 |
| per Block: `ln_1` 1,536 + `c_attn` 1,771,776 + attn `c_proj` 590,592 + `ln_2` 1,536 + `c_fc` 2,362,368 + mlp `c_proj` 2,360,064 | 7,087,872 |
| 12 Blocks | 85,054,464 |
| `ln_f` | 1,536 |
| `lm_head` | 0 (tied) |
| **Total** | **124,475,904** |

`configure_optimizers` prints the split it uses for weight decay: **50 tensors / 124,354,560 params decayed** (all 2D: embeddings and matmul weights), and **98 tensors / 121,344 params not decayed** (biases and LayerNorm).

### Initialization (`_init_weights`, applied to every submodule)

| What | Init |
|---|---|
| `nn.Linear.weight` | `N(0, 0.02)` |
| `nn.Linear.weight` with `NANOGPT_SCALE_INIT` (attn `c_proj`, mlp `c_proj`) | `N(0, 0.02 / sqrt(2 · n_layer))` = `N(0, 0.00408)` for 12 layers |
| `nn.Linear.bias` | zeros |
| `nn.Embedding.weight` (`wte`, `wpe`) | `N(0, 0.02)` |
| `nn.LayerNorm` | PyTorch default (weight 1, bias 0) |

The `c_proj` scaling makes up for the `2·n_layer` residual additions, so the residual stream's variance doesn't grow with depth. Because `wte` and `lm_head` are the same tensor, it gets initialized twice (once as a Linear, once as an Embedding). Both use std 0.02, so the result is the same.

### `GPT.from_pretrained(model_type)`

Builds a `GPT` with the matching config (vocab 50257), loads HF `GPT2LMHeadModel`, and copies every tensor by name. OpenAI used `Conv1D`, which stores weights as `(in, out)`, so `c_attn`, `c_proj` and `c_fc` weights are transposed to `nn.Linear`'s `(out, in)`. Keys ending in `.attn.bias` and `.attn.masked_bias` (mask buffers) are skipped.

### `GPT.configure_optimizers(weight_decay, learning_rate, device_type, master_process=True)`

AdamW with `betas=(0.9, 0.95)`, `eps=1e-8`, two parameter groups (decay / no decay, split by `p.dim() >= 2`), and `fused=True` when available and `device_type.startswith("cuda")`.

## 2. Data pipeline

### On disk (produced by `datasets/fineWeb/fineweb.py`)

```
datasets/fineWeb/edu_fineweb10B/
  edufineweb_val_000000.npy     100M uint16 tokens   ← only val shard
  edufineweb_train_000001.npy   100M uint16 tokens
  ...
  edufineweb_train_0000NN.npy   remainder
```

Each document is `<|endoftext|>` followed by its tokens. Documents are packed back to back with no padding and can span two shards.

### `DataLoaderLite(B, T, process_rank, num_processes, split, master_process=True)`

- Picks shards by substring match on `split` (`'train'` or `'val'`), sorted.
- Holds **one shard in memory** at a time, as an `int64` tensor (100M × 8 bytes ≈ 800MB per loader per rank).
- `next_batch()` slices `B·T + 1` tokens from `current_position`: `x = buf[:-1].view(B, T)`, `y = buf[1:].view(B, T)`. Targets are inputs shifted by one.
- **DDP striding:** rank *r* starts at `r·B·T` and every rank advances by `B·T·world_size` each call, so in each "round" the ranks take consecutive, non-overlapping chunks:

  ```
  shard tokens: | r0 | r1 | r2 | ... | rN-1 | r0 | r1 | ... |
                 ◄B·T►
  ```

- When the next round wouldn't fit, it loads the next shard (`% len(shards)`, so it wraps) and goes back to offset `r·B·T`. The few leftover tokens at the end of a shard are skipped.
- `reset()` goes back to shard 0, offset `r·B·T`. The val loader is reset before every val eval, so it always sees the same tokens.

## 3. Training (`train.py`)

### Setup in `main()`

1. `setup_distributed()` → `ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process`.
2. Seed 1337 (CPU + CUDA or MPS). Under DDP every rank uses the same seed, so they start from identical weights. DDP also broadcasts rank 0's weights when it wraps the model.
3. `grad_accum_steps = total_batch_size // (B · T · world_size)`.
4. Train and val `DataLoaderLite`s.
5. `torch.set_float32_matmul_precision('high')` (TF32).
6. `GPT(GPTConfig(vocab_size=50304))` → `.to(device)` → `torch.compile` (if `use_compile`) → `DDP(...)` (if ddp). `raw_model` = the unwrapped `GPT`.
7. `log/` is created and `log/log.txt` truncated.
8. `optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type, ...)`.

### One step

```
for step in range(max_steps):
    ┌ if step % eval_interval == 0 or last step:
    │     val loss (20 batches)        → log "{step} val {x}"
    │     checkpoint if step > 0 and (step % checkpoint_interval == 0 or last step)
    │ if same condition and not use_compile:
    │     HellaSwag                    → log "{step} hella {x}"
    │ if step > 0 (or last) and same interval and not use_compile:
    └     print 4 samples

    model.train(); optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() → to(device)
        if ddp: model.require_backward_grad_sync = (micro_step == last)   ← BEFORE forward
        with autocast(device_type, bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()                                  # grads accumulate
    if ddp: all_reduce(loss_accum, AVG)                  # for logging only
    norm = clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step); set on every param group
    optimizer.step()
    torch.cuda.synchronize()                              # accurate timing (CUDA-only!)
    print/log "{step} train {loss_accum}"
```

### Learning-rate schedule

```
lr
6e-4 ┤      ╭─╮
     │     ╱   ╲___
     │    ╱        ╲___
     │   ╱             ╲____
6e-5 ┤  ╱                   ╲________
     └──┴───────────────────────────┴──► step
       0  715                      19073
       warmup        cosine decay
```

- `step < 715`: `max_lr · (step + 1) / 715` (linear).
- `715 ≤ step ≤ 19073`: `min_lr + 0.5·(1 + cos(π · progress))·(max_lr − min_lr)`.
- `step > 19073`: `min_lr` (never reached with the current `max_steps`).

### Numbers per optimizer step

| | 1 GPU | 8 GPUs |
|---|---|---|
| tokens per micro-batch per GPU | 16 × 1024 = 16,384 | 16,384 |
| `grad_accum_steps` | 32 | 4 |
| tokens per optimizer step | 524,288 | 524,288 |
| steps for ~10B tokens | 19,073 | 19,073 |

## 4. Evaluation (`evaluation.py`)

| Function | What it does | Returns |
|---|---|---|
| `estimate_val_loss(model, val_loader, device, device_type, ddp, val_loss_steps=20)` | `model.eval()`, `val_loader.reset()`, mean loss over 20 batches under `no_grad` + autocast, AVG all-reduced | `val_loss_accum` (0-dim tensor) |
| `evaluate_hellaswag(model, device, device_type, ddp, ddp_rank, ddp_world_size)` | Loops over all 10,042 val examples, keeping those with `i % world_size == rank`. For each, forward the `4 × N` tokens and `get_most_likely_row`. SUM all-reduces counts | `(num_correct_norm, num_total, acc_norm)` |
| `generate_samples(model, device, device_type, ddp_rank, num_return_sequences=4, max_length=32, prompt=...)` | `model.eval()`, top-50 sampling with a private `torch.Generator` seeded `42 + rank`, prints each sample | `None` |
| `get_most_likely_row(tokens, mask, logits)` | Per-token CE (shifted by one), averaged over the masked ending tokens, `argmin` over the 4 rows | predicted index |

`evaluate_hellaswag` does **not** call `model.eval()` itself. It relies on `estimate_val_loss` having just run on the same step. Since GPT-2 has no dropout or batch-norm, train and eval mode produce the same numbers anyway.

## 5. Distributed (`distributed.py`)

```python
@dataclass
class DistributedContext:
    ddp: bool             # True when launched by torchrun (RANK env var present)
    ddp_rank: int         # global rank
    ddp_local_rank: int   # GPU index on this node
    ddp_world_size: int
    device: str           # 'cuda:N' under torchrun; 'cuda' / 'mps' / 'cpu' otherwise
    device_type: str      # 'cuda' / 'mps' / 'cpu', what torch.autocast wants
    master_process: bool  # rank 0: the only one that prints, logs, checkpoints
```

DDP requires CUDA (`backend='nccl'`). `cleanup_distributed(ddp)` calls `destroy_process_group()` when needed.

## 6. Outputs

| File | Written by | Format |
|---|---|---|
| `log/log.txt` | master, every step + every eval | `{step} train {loss:.6f}` / `{step} val {loss:.4f}` / `{step} hella {acc:.4f}` |
| `log/model_{step:05d}.pt` | master, steps 5000, 10000, 15000 and 19072 | `{'model': state_dict, 'config': GPTConfig, 'step': int, 'val_loss': float}` |

## 7. Gotchas

- **`device` vs `device_type`.** Under torchrun `device` is `'cuda:N'`. Anything that compares to `"cuda"` must use `.startswith("cuda")` or `device_type`. `torch.autocast` rejects `'cuda:0'`.
- **Set `require_backward_grad_sync` before the forward pass.** DDP reads it inside `forward()`. Setting it only before `backward()` does nothing, and gradients sync on every micro-step.
- **Wrapping order: `torch.compile` first, then `DDP`.** Otherwise `require_backward_grad_sync` lands on the compile wrapper. Always use `raw_model` for `configure_optimizers` and `state_dict()`.
- **`use_compile = True` skips HellaSwag and sampling** (recompiles on varying shapes). It is currently `False`.
- **`datasets/` shadows HuggingFace `datasets`.** `evaluation.py` appends `datasets/hellaSwag` to `sys.path` and imports `hellaswag` directly. Don't change it to `from datasets.hellaSwag import ...`.
- **`from_pretrained` filters `.attn.bias`** even though the mask buffer is no longer registered (it is commented out along with the naive attention).
- **`vocab_size`:** 50304 for training, 50257 for `from_pretrained`. A 50304 checkpoint can't be loaded into a 50257 model or the reverse.
- **No resume:** `log.txt` is truncated at startup, and checkpoints have no optimizer or RNG state.
- **CUDA-only loop:** `torch.cuda.synchronize()` is unconditional.
- **Run from the repo root:** the shard path is relative to the current directory.
- **The caller moves tensors to the device** (`x.to(device)`). The model takes no device argument.
