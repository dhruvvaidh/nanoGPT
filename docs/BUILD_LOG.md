# Build log: how this project was built, step by step

This is a chronological record of how the project went from an empty repo to a GPT-2 (124M) trained on 10B tokens. Each step lists the commit(s), what changed in the code, *why* (the concept behind it), and any bugs that came up. Commit hashes are short SHAs on `main`, so `git show <sha>` shows the exact diff.

For how the finished code fits together, see [`ARCHITECTURE.md`](ARCHITECTURE.md). For how to run it, see the [README](../README.md).

## At a glance

| # | Step | Commit(s) | Date |
|---|---|---|---|
| 0 | [Initial commit](#step-0--initial-commit) | `92ff6a9` | Sep 14 |
| 1 | [GPT-2 model + loading OpenAI's weights](#step-1--gpt-2-model--loading-openais-weights) | `5c8f669` | Sep 19 |
| 2 | [Random init sanity check](#step-2--random-init-sanity-check) | `e8c93cd` | Sep 19 |
| 3 | [Loss, optimizer, overfitting one batch](#step-3--loss-optimizer-overfitting-one-batch) | `6a7f552` | Sep 19 |
| 4 | [`DataLoaderLite`](#step-4--dataloaderlite) | `d19821f` | Sep 19 |
| 5 | [Weight tying](#step-5--weight-tying) | `79bcb0c` | Sep 19 |
| 6 | [GPT-2 weight initialization](#step-6--gpt-2-weight-initialization) | `ccf1238`, `28cf555` | Sep 19 |
| 7 | [Optimization 1: TF32 matmuls](#step-7--optimization-1-tf32-matmuls) | `2937a14` | Sep 19 |
| 8 | [Optimization 2: bfloat16 autocast](#step-8--optimization-2-bfloat16-autocast) | `bed9b91` | Sep 19 |
| 9 | [Optimization 3: `torch.compile`](#step-9--optimization-3-torchcompile) | `93170f8`, `dc6577b` | Sep 20 |
| 10 | [Optimization 4: Flash attention](#step-10--optimization-4-flash-attention) | `62cca4c` | Sep 20 |
| 11 | [Optimization 5: "Nice" numbers](#step-11--optimization-5-nice-numbers) | `e7b803a` | Sep 20 |
| 12 | [Tuning 1: AdamW betas + gradient clipping](#step-12--tuning-1-adamw-betas--gradient-clipping) | `371ad9c` | Sep 20 |
| 13 | [Tuning 2: LR schedule, weight decay, fused AdamW](#step-13--tuning-2-lr-schedule-weight-decay-fused-adamw) | `7b9be23` | Sep 20 |
| 14 | [Gradient accumulation (0.5M-token batches)](#step-14--gradient-accumulation-05m-token-batches) | `3eb638a` | Sep 20 |
| 15 | [Distributed Data Parallel (multi-GPU)](#step-15--distributed-data-parallel-multi-gpu) | `269cc7f`, `3fe0dde` | Sep 20 |
| 16 | [FineWeb-Edu 10B dataset](#step-16--fineweb-edu-10b-dataset) | `b6375a8` | Sep 20 |
| 17 | [Validation split, logging, checkpoints](#step-17--validation-split-logging-checkpoints) | `b6375a8` | Sep 20 |
| 18 | [HellaSwag evaluation](#step-18--hellaswag-evaluation) | `b6375a8` | Sep 20 |
| 19 | [In-training sampling and the full run](#step-19--in-training-sampling-and-the-full-run) | `b6375a8` | Sep 20 |
| 20 | [Modularize the codebase](#step-20--modularize-the-codebase) | `61c5075`, `642924a` | Sep 23–24 |

Then: a [summary of the bugs found along the way](#bugs-found-along-the-way), and [what was different during the full training run](#what-the-full-run-actually-used).

---

## Phase 1: The model

### Step 0 — Initial commit

**Commit:** `92ff6a9`

Only GitHub's standard Python `.gitignore`, 218 lines. Later steps add `CLAUDE.md`, `edu_fineweb10B/` and `log/` to it.

### Step 1 — GPT-2 model + loading OpenAI's weights

**Commit:** `5c8f669`

Creates `train.py` (198 lines) with the whole GPT-2 architecture. The module and parameter names match HuggingFace's `GPT2LMHeadModel` exactly, so its state dict can be copied straight in.

- **`GPTConfig`** (dataclass): `block_size=1024`, `vocab_size=50257` (50,000 BPE merges + 256 byte tokens + `<|endoftext|>`), `n_layer=12`, `n_head=12`, `n_embd=768`.
- **`CausalSelfAttention`**: one `c_attn` Linear produces Q, K and V for all heads at once (`n_embd → 3·n_embd`). The result is split and reshaped to `(B, n_head, T, head_size)`, run through `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, re-assembled to `(B, T, C)`, then passed through the output projection `c_proj`.
- **`MLP`**: `c_fc` (768 → 3072), `GELU(approximate='tanh')`, `c_proj` (3072 → 768). The tanh approximation is what GPT-2 used. The exact GELU was slow in TensorFlow at the time, and the approximation became standard.
- **`Block`**: pre-LayerNorm residual block, `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))`. The comment captures the intuition: **attention is a "reduce"** (tokens exchange information, a weighted pooling over the sequence). **The MLP is a "map"** (each token is processed on its own).
- **`GPT`**: `wte` (token embedding), `wpe` (learned position embedding), 12 `Block`s, a final `ln_f`, and `lm_head` (768 → vocab, no bias).
- **`GPT.from_pretrained(model_type)`**: builds the right config for `gpt2` / `-medium` / `-large` / `-xl`, loads the HF model, and copies every tensor across. OpenAI's checkpoints use a `Conv1D` module whose weights are stored transposed relative to `nn.Linear`. So `attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` weights are `.t()`'d on the way in. Buffers ending in `.attn.bias` / `.attn.masked_bias` (the causal mask) are filtered out on both sides.
- **Generation script**: encodes `"Hello, I'm a language model,"` with tiktoken, makes 5 copies, and samples up to 30 tokens. Each step takes the last position's logits, applies softmax, keeps the **top-k = 50** (what the HF `pipeline` uses), samples with `torch.multinomial`, `gather`s the token id and appends it.

Device was hardcoded to `mps` if available, otherwise `cpu`.

> Note: Flash attention (`scaled_dot_product_attention`) was used **from this very first commit**, even though "Optimization 4" (Step 10) is named after it.

### Step 2 — Random init sanity check

**Commit:** `e8c93cd`

- Comments out `GPT.from_pretrained('gpt2')` and uses `GPT(GPTConfig())` instead. The model still runs, but it produces gibberish, which shows the pipeline works without OpenAI's weights.
- Device auto-detection: `cuda` > `mps` > `cpu`.

### Step 3 — Loss, optimizer, overfitting one batch

**Commit:** `6a7f552`

- `GPT.forward(idx, targets=None)` now returns `(logits, loss)`. Loss is `F.cross_entropy` on logits flattened to `(B·T, vocab)` and targets flattened to `(B·T,)`.
- Removed the global `device` from inside the model. Position indices are created on `idx.device`, so the model has no device dependency and the caller moves tensors to the device.
- **Data:** added `datasets/tiny_shakespeare/input.txt`. Takes the first 1,000 characters, tokenizes them, and builds one batch with `B, T = 4, 32`:
  ```python
  buf = torch.tensor(tokens[:B*T+1])
  x = buf[:-1].view(B, T)   # inputs
  y = buf[1:].view(B, T)    # targets: every token's label is the next token
  ```
- **Training:** `AdamW(lr=3e-4)`, 50 steps on that *same* batch. Overfitting a single batch is the classic sanity check: if the loss doesn't go to ~0, something is broken. The expected starting loss for a random model is about `-ln(1/50257) ≈ 10.8` (uniform over the vocab).
- The generation code was commented out. It still sits at the bottom of `train.py` today.
- **Notebook** `notebooks/walkthrough.ipynb` added. It loads HF GPT-2 and prints every tensor shape, plots `wpe` as an image and some of its columns as curves (learned position embeddings come out as smooth, sinusoid-like curves), plots a block of `c_attn` weights, runs the HF `pipeline` generator, does manual sampling, and shows how the `x`/`y` batch is built from a token buffer.

### Step 4 — `DataLoaderLite`

**Commit:** `d19821f`

```python
class DataLoaderLite:
    def __init__(self, B, T): ...      # tokenizes all of input.txt once
    def next_batch(self): ...          # returns (x, y), advances by B*T
```

It walks through the whole token stream (~338K tokens for Tiny Shakespeare) instead of reusing one batch.

**Bug:** the end-of-data check was `if self.current_position > self.current_position + B*T + 1`, which is never true. So the loader would eventually run off the end of the data. It was fixed in Step 7.

### Step 5 — Weight tying

**Commit:** `79bcb0c`

```python
self.transformer.wte.weight = self.lm_head.weight
```

The token embedding (`wte`, maps id → vector) and the output classifier (`lm_head`, maps vector → logits over ids) share **one** `(vocab, n_embd)` matrix. The notebook explains why, with a new markdown cell and cells checking that HF's two tensors are equal and share the same `data_ptr()`:

- *Attention Is All You Need* §3.4 shares this matrix. The idea comes from Press & Wolf, *Using the Output Embedding to Improve Language Models*.
- Intuition: tokens that are semantically similar should be close in the input embedding *and* get similar output probabilities, so one matrix can serve both jobs.
- It saves 50257 × 768 ≈ 38.6M parameters, about 30% of the 124M model.

Also fixed a print that dumped the whole token list instead of `len(tokens)`.

### Step 6 — GPT-2 weight initialization

**Commits:** `ccf1238`, `28cf555` (bug fix)

`self.apply(self._init_weights)` walks every submodule:

| Module | Init |
|---|---|
| `nn.Linear` weight | `normal(0, 0.02)` |
| `nn.Linear` weight on a residual projection (`c_proj` in attention **and** MLP, marked with `NANOGPT_SCALE_INIT = 1`) | `normal(0, 0.02 · (2·n_layer)^-0.5)` |
| `nn.Embedding` (wte, wpe) | `normal(0, 0.02)` |

The values come from OpenAI's GPT-2 code. 0.02 is in the same range as `1/sqrt(n_embd)` for GPT-2's model widths (0.036 at 768, 0.025 at 1600). The extra scaling for residual projections is from the GPT-2 paper. Every block adds its output to the residual stream twice (once from attention, once from the MLP), so with `n_layer` blocks there are `2·n_layer` additions. Each addition adds variance, so the standard deviation of the stream grows like `sqrt(N)`. Scaling each contribution by `N^-0.5` keeps it near 1. The notebook demonstrates this: summing 100 random vectors scaled by `100**-0.5` gives a std of ≈ 0.985 instead of ≈ 10.

Also added seeding (`torch.manual_seed(1337)` plus the CUDA/MPS equivalents).

**Bugs, fixed in `28cf555`:** `self.config.n_layers` → `n_layer` (the config field has no `s`), and `hasattr(torch.backends)` → `hasattr(torch.backends, 'mps')` (`hasattr` needs two arguments).

(Linear biases were *not* zeroed here. That was added much later, in Step 20.)

---

## Phase 2: Making it fast

From here on, each step prints `dt` (ms per step) and `tokens/sec`, so speedups can be measured. Training moves to a CUDA GPU.

### Step 7 — Optimization 1: TF32 matmuls

**Commit:** `2937a14`

```python
torch.set_float32_matmul_precision('high')
```

- By default (`'highest'`), fp32 matmuls are done in full fp32: 8 exponent bits and 23 stored mantissa bits.
- `'high'` lets PyTorch use **TensorFloat-32** on Ampere+ tensor cores. TF32 keeps fp32's 8-bit exponent (same range) but only 10 mantissa bits. The inputs to each multiply are rounded, and the accumulation is still fp32. In code, tensors are still fp32. Only the internal matmul gets cheaper, with a much higher theoretical throughput.
- Deep learning tolerates the lost precision easily. In practice the speedup is smaller than the theoretical number, because the model is partly **memory-bandwidth bound** (data still moves around as full fp32).

Also in this commit:
- Micro-batch increased to `B=16, T=1024` (the real GPT-2 context length).
- Timing: `t0`/`t1` around each step, plus **`torch.cuda.synchronize()`**. CUDA calls are asynchronous: Python queues the kernels and continues. Without the synchronize, you would measure how fast the CPU queues work, not how fast the GPU does it. (This unconditional CUDA call is why the loop doesn't run on Mac or CPU today.)
- Fixed the `DataLoaderLite` wrap bug from Step 4: reset when `current_position + B*T + 1 > len(tokens)`.

### Step 8 — Optimization 2: bfloat16 autocast

**Commit:** `bed9b91`

```python
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

- **Mixed precision:** inside the autocast region, operations that are safe in low precision (matmuls, the Linear layers) run in bf16. Precision-sensitive operations (softmax, LayerNorm, the loss) stay in fp32. The list of which ops get cast is in the PyTorch autocast docs. Parameters and gradients stay fp32; only the activations change.
- **Why bf16 and not fp16:** bf16 has the *same 8-bit exponent as fp32* (same range) with fewer mantissa bits. fp16 has only a 5-bit exponent, so small gradients underflow to zero and you need a `GradScaler` to scale the loss up and back down. bf16 doesn't need that.
- Following the PyTorch docs, only the **forward pass and loss** are wrapped. `backward()` and `optimizer.step()` stay outside.

### Step 9 — Optimization 3: `torch.compile`

**Commits:** `93170f8`, `dc6577b` (comment addition)

```python
model = torch.compile(model)
```

The comments name two advantages:

1. **No Python interpreter overhead.** Eager PyTorch runs the `forward` one line at a time, launching one kernel per operation. `torch.compile` traces the whole forward pass ahead of time and turns it into an optimized program that Python doesn't re-interpret on each call.
2. **Kernel fusion (fewer HBM round trips).** In eager mode each operation (e.g. every term inside the tanh-GELU) reads its input from GPU memory (HBM) and writes the result back. When the compiler can see the whole sequence of operations, it fuses them into one kernel that keeps intermediate values in on-chip registers/SRAM and only touches HBM once. Memory traffic, not arithmetic, is usually the bottleneck.

The first step is slow because compilation happens then. (In Step 15, `torch.compile` was put behind `use_compile = False`, and it is still off.)

### Step 10 — Optimization 4: Flash attention

**Commit:** `62cca4c`

This commit **only added comments**. The model was already calling `F.scaled_dot_product_attention(..., is_causal=True)` from Step 1. It added, commented out, the manual "naive" attention that flash attention replaces, for reference:

```python
# self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
# att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
# att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
# att = F.softmax(att, dim=-1)
# y = att @ v
```

Why flash attention is faster: the naive version materializes the `(B, n_head, T, T)` attention matrix in HBM, then reads it back for the mask, again for the softmax, and again for `@ v`. FlashAttention computes attention in tiles that fit in on-chip SRAM, using an **online softmax** (running max and sum) so the full `T×T` matrix never exists in HBM. It actually does *more* FLOPs, but far less memory traffic, which makes it much faster. `torch.compile` can't discover this fusion on its own, because it needs the algorithm rewritten.

The `bias` buffer name is also why `from_pretrained` filters keys ending in `.attn.bias`.

### Step 11 — Optimization 5: "Nice" numbers

**Commit:** `e7b803a`

```python
model = GPT(GPTConfig(vocab_size=50304))
```

CUDA kernels work in tiles whose sizes are powers of two (32, 64, 128 …). A dimension like 50257 (odd, prime-ish) leaves a ragged last tile that is handled by slower boundary code. 50304 = 393 × 128 divides evenly by every power of two up to 128.

The 47 extra tokens never appear in the data, so the model just learns to push their logits toward −∞. The extra compute is negligible, and the per-step time *drops* despite doing slightly more work. This is why training uses `vocab_size=50304` while `from_pretrained` still uses 50257.

---

## Phase 3: GPT-3 training hyperparameters

The GPT-2 paper barely describes its training setup. The GPT-3 paper describes it in detail, and its "Small" model is about the same size, so its settings are used.

### Step 12 — Tuning 1: AdamW betas + gradient clipping

**Commit:** `371ad9c`

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
...
loss.backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

- **β₂ = 0.95** (PyTorch's default is 0.999): the second-moment estimate adapts faster, which GPT-3 found more stable for large models.
- **Global gradient-norm clipping at 1.0:** all gradients are treated as one long vector, its L2 norm is computed, and if the norm is above 1.0 everything is scaled down to norm 1.0. An unlucky batch can produce a huge loss and a huge gradient that "shocks" the model. Clipping caps the size of the update.
- The norm is printed every step. It is a useful health signal: it should settle down after the first few steps, and spikes or a steady climb mean instability.

### Step 13 — Tuning 2: LR schedule, weight decay, fused AdamW

**Commit:** `7b9be23`

**Learning-rate schedule** (`get_lr`): linear warmup, then cosine decay to 10% of the peak:

```python
max_lr = 6e-4; min_lr = max_lr * 0.1
def get_lr(it):
    if it < warmup_steps:   return max_lr * (it + 1) / warmup_steps   # +1 so step 0 isn't lr=0
    if it > max_steps:      return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))           # 1 → 0
    return min_lr + coeff * (max_lr - min_lr)
```

It is applied by hand each step: `for g in optimizer.param_groups: g['lr'] = lr`. At this point `warmup_steps=10, max_steps=50` (toy values). Step 16 sets them to the real ones. `6e-4` is GPT-3 Small's peak LR.

**`GPT.configure_optimizers(weight_decay, learning_rate, device_type)`** builds two parameter groups:
- **Decayed** (`dim >= 2`): every matmul weight and the embeddings, 50 tensors, 124,354,560 parameters.
- **Not decayed** (`dim < 2`): biases and LayerNorm gains/biases, 98 tensors, 121,344 parameters.

Weight decay 0.1 (as in GPT-3) pulls large weights toward zero. This is a regularizer that pushes the network to spread its work across many weights. Decaying a bias or a LayerNorm scale has no such benefit.

**Fused AdamW** (`fused=True` when the device is CUDA and the installed PyTorch supports it, checked with `inspect.signature`): instead of launching a separate set of kernels for each of the ~150 parameter tensors, the whole AdamW update runs as one fused kernel. This is the same kernel-fusion idea as Step 9, applied to the optimizer.

The step log becomes `step | loss | lr | norm | dt | tokens/sec`.

### Step 14 — Gradient accumulation (0.5M-token batches)

**Commit:** `3eb638a`

GPT-3 Small trains with a batch of **~0.5M tokens**. `B=16, T=1024` is only 16,384 tokens, so the larger batch is simulated by running several micro-batches and summing their gradients before one optimizer step:

```python
total_batch_size = 524288              # 2**19 — a "nice" number close to 0.5M
grad_accum_steps = total_batch_size // (B * T)   # = 32 on one GPU

optimizer.zero_grad()
loss_accum = 0.0
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    with torch.autocast(...):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps     # <-- critical
    loss_accum += loss.detach()
    loss.backward()                    # gradients ADD UP across micro-steps
```

**Why divide by `grad_accum_steps`:** `cross_entropy` returns the *mean* over its B·T tokens. Calling `.backward()` repeatedly *sums* gradients. Summing 32 means gives 32× the gradient of the mean over the full batch. Dividing each micro-loss restores the "mean over all 524,288 tokens" objective. The notebook proves this on a tiny MLP: one batch of 4 gives exactly the same gradient as 4 batches of 1 with `loss / 4`.

`loss_accum` (detached) is only for logging. `tokens/sec` now counts `B·T·grad_accum_steps`.

### Step 15 — Distributed Data Parallel (multi-GPU)

**Commits:** `269cc7f`, `3fe0dde`

Launching with `torchrun --standalone --nproc_per_node=8 train.py` starts **8 copies** of the script, one per GPU, with `RANK`, `LOCAL_RANK` and `WORLD_SIZE` set as environment variables.

- **Setup:** if `RANK` is set, call `init_process_group(backend='nccl')`, pick `device = f'cuda:{LOCAL_RANK}'`, and set `master_process = (rank == 0)`. Only the master process prints and (later) logs and writes checkpoints. Otherwise it is a normal single-device run with `world_size=1`.
- **Data:** `DataLoaderLite(B, T, process_rank, num_processes)`. Rank `r` starts at offset `r·B·T` and every rank advances by `B·T·world_size`, so the ranks read interleaved chunks that never overlap.
- **Batch math:** `grad_accum_steps = total_batch_size // (B·T·world_size)`. With 8 GPUs that is 4 micro-steps; together they still add up to 524,288 tokens per step.
- **Wrapping:** `model = DDP(model, device_ids=[local_rank])`. During `backward()`, DDP **averages gradients across all ranks** with all-reduce. It does this in buckets, overlapping the communication with the rest of the backward pass, so after the backward pass every rank has identical gradients and takes an identical optimizer step.
- **`raw_model = model.module if ddp else model`.** The DDP wrapper doesn't expose `GPT`'s own methods, so `configure_optimizers` (and later, checkpointing) uses the unwrapped model.
- **Sync only on the last micro-step:** `model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)`. Syncing every micro-step would just all-reduce partial sums 4× for nothing.
- **Logging:** `dist.all_reduce(loss_accum, op=AVG)` so the printed loss is the mean over all ranks, not just rank 0's share. `tokens/sec` is multiplied by `world_size`.
- `use_compile = False` was introduced, with compile placed *before* the DDP wrap.
- `configure_optimizers` now uses `device_type.startswith("cuda")`, because under torchrun `device` is `'cuda:0'`, not `'cuda'`.
- `destroy_process_group()` at the end.

**Latent issues in this commit, fixed later:**
1. `require_backward_grad_sync` was set **after** the forward pass. DDP reads the flag *inside* `forward()` to decide whether to arm its backward hooks, so it had no effect and gradients synced on every micro-step. That is wasted communication but still correct gradients. Fixed in Step 20.
2. `torch.autocast(device_type=device)` received `'cuda:N'` under torchrun. Fixed in Step 16 with `device_type`.

`3fe0dde` ("removing claude.md file") added `CLAUDE.md` to `.gitignore`. The file had already been committed, so it stayed tracked and the ignore entry has no effect.

---

## Phase 4: Real data, evals, the full run

All of Steps 16–19 landed together in `b6375a8`.

### Step 16 — FineWeb-Edu 10B dataset

Tiny Shakespeare is ~338K tokens: fine for debugging, far too small for pretraining. **FineWeb-Edu** is web text filtered for educational quality, and its `sample-10BT` subset gives roughly 10B GPT-2 tokens.

**`datasets/fineWeb/fineweb.py`** (run once, offline):
1. `load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")`.
2. Tokenizes documents in parallel with `multiprocessing.Pool(cpu_count // 2)`. Each document becomes `[<|endoftext|>] + enc.encode_ordinary(text)`, so the EOT token *separates* documents and the model learns where one ends and the next begins.
3. Stores tokens as **`uint16`** (asserting every id < 2¹⁶), which halves the disk size compared with int32.
4. Fills a fixed buffer of `shard_size = 100M` tokens. When a document doesn't fit, it is split: the head finishes the current shard and the tail starts the next one.
5. Writes `edufineweb_{split}_{NNNNNN}.npy`. **Shard 0 is `val`**, all others are `train`.

**`train.py` changes:**
- `load_tokens(filename)`: `np.load`, then `int32`, then `torch.long` (a `uint16` array can't be turned into a long tensor directly).
- `DataLoaderLite(..., split)`: lists `datasets/fineWeb/edu_fineweb10B`, keeps files whose name contains `split`, sorts them, and loads shard 0. When the next batch would overrun the current shard, it moves to the next shard (wrapping with `% len(shards)`) and resets to the rank's offset. `reset()` goes back to shard 0.
- Real schedule: **`max_steps = 19073`** (10B tokens ÷ 524,288 tokens/step ≈ 1 epoch) and **`warmup_steps = 715`** (GPT-3 warms up over 375M tokens, and 375M ÷ 524,288 ≈ 715).
- `device_type = "cuda" if device.startswith("cuda") else device`, now passed to every `torch.autocast`.
- Comment on `B = 16`: B=64 runs out of memory on a 40GB A100 and needs an 80GB card.

### Step 17 — Validation split, logging, checkpoints

Every 250 steps and on the last step:

- **Validation loss:** `model.eval()`, `val_loader.reset()` (so it is the *same* val tokens every time and the numbers are comparable), then 20 batches under `torch.no_grad()` and autocast, averaged, and all-reduced across ranks. That is 20 × 16 × 1024 ≈ 328K tokens per rank. Validation loss is what tells you whether the model generalizes or is memorizing the training data.
- **`log/log.txt`:** truncated at startup, then appended as `{step} val {loss}` here and `{step} train {loss}` every step.
- **Checkpoints:** when `step > 0 and (step % 5000 == 0 or last_step)`, the master process saves `log/model_{step:05d}.pt` = `{'model': raw_model.state_dict(), 'config': raw_model.config, 'step': step, 'val_loss': ...}`. No optimizer or RNG state is saved, so these are for inference, not for resuming.
- `model.train()` is called before each step's accumulation loop, because the evals put the model in eval mode. (GPT-2 has no dropout, so the mode doesn't change any numbers today, but it is correct practice.)

### Step 18 — HellaSwag evaluation

Validation loss only tells you how well the model predicts FineWeb text. **HellaSwag** is an external common-sense benchmark: given a context, pick the most plausible of 4 endings. Wrong endings were generated adversarially, so they are tricky for language models.

**`datasets/hellaSwag/hellaswag.py`** (adapted from Karpathy's version):
- The original GitHub download is blocked, so it fetches the **HuggingFace parquet mirror** (`Rowan/hellaswag`) and converts it to the jsonl format the code expects (string label → `int`, endings → `list`). It writes to a temp file and renames it, raises on HTTP errors, and checks that a cached file starts with a JSON object, so a saved "404" page isn't mistaken for data.
- `render_example`: builds a `4 × N` token tensor (context + `" " + ending` for each candidate, zero-padded) and a **mask** that is 1 only on the ending's tokens.
- `evaluate(model_type, device)` + CLI (`-m gpt2`, `-d cuda`): scores HF GPT-2 as a reference.

**Scoring, "completion style"** (`get_most_likely_row`): instead of asking the model "A, B, C or D?", which a 124M model can't do, the model reads each of the 4 full sequences. Per-token cross-entropy is computed at each position, logits shifted by one, and the mask shifted the same way. The loss is averaged **over the ending's tokens only**, and the ending with the **lowest average loss** is the prediction. Averaging (instead of summing) stops longer endings from being penalized. This is the `acc_norm` metric. Random chance is 25%.

**In the loop:** every 250 steps (skipped when `use_compile`, because the varying sequence lengths trigger recompiles). Examples are sharded across ranks (`i % world_size == rank`), correct/total counts are all-reduced with SUM, and `{step} hella {acc}` is logged.

`datasets/` collides with the HuggingFace `datasets` package (which `fineweb.py` imports), so `from datasets.hellaSwag import ...` would resolve to the wrong package. Instead, `datasets/hellaSwag` is appended to `sys.path` and `hellaswag` is imported as a top-level module.

### Step 19 — In-training sampling and the full run

Every 250 steps (not step 0, and not when compiled), the model generates 4 continuations of `"Hello, I'm a language model,"` up to 32 tokens, using the same top-50 sampling as Step 1 and printing `rank {r} sample {i}: ...`. It uses its **own `torch.Generator` seeded with `42 + rank`**, so sampling doesn't consume the global RNG that training uses, and each rank produces different samples.

**The run:** with all of this in place, the full 19,073-step run was done on a rented CUDA machine (Lambda) with `torchrun`. Final result: **val loss ≈ 3.40, HellaSwag `acc_norm` ≈ 0.27**. OpenAI's GPT-2 124M scores 3.29 val loss on this val shard and 0.2955 on HellaSwag with this script. `log/log.txt` and the `log/model_*.pt` checkpoints from the run are kept locally (gitignored).

---

## Phase 5: Cleanup

### Step 20 — Modularize the codebase

**Commits:** `61c5075`, merged via PR #1 as `642924a`

By now `train.py` was 633 lines holding the model, the data loader, the eval helpers and the training script, all executing at import time. It was split into five flat modules:

| Module | Contents |
|---|---|
| `model.py` | `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, `GPT` (+ `from_pretrained`, `configure_optimizers`) |
| `dataloader.py` | `load_tokens`, `DataLoaderLite` |
| `evaluation.py` | `estimate_val_loss`, `evaluate_hellaswag`, `generate_samples`, `get_most_likely_row`, shared `enc`, the `sys.path` hack |
| `distributed.py` | `setup_distributed() -> DistributedContext`, `cleanup_distributed()` |
| `train.py` | hyperparameters, `get_lr`, `main()` behind `if __name__ == "__main__"` |

**Structural changes:**
- `master_process` is now an explicit argument (default `True`) to `configure_optimizers` and `DataLoaderLite`, instead of a global read from the training script. That global had tied them to one file.
- Each eval returns its numbers. `main()` is the only place that prints, logs and checkpoints.
- `configure_optimizers` is passed `device_type` (`'cuda'`) instead of `device` (`'cuda:0'`). The result is the same, because the check uses `.startswith`.
- The `250` and `5000` intervals became `eval_interval` and `checkpoint_interval`.
- Every explanatory comment, and every commented-out alternative, moved with its code (all 217 comment tokens were checked).

**Behavior changes carried in:**
- **Linear biases are zero-initialized** (`torch.nn.init.zeros_`), matching GPT-2. PyTorch's default is a small uniform init.
- **`require_backward_grad_sync` is set *before* the forward pass**, fixing the Step 15 issue. Gradients now sync only on the last micro-step.
- `.gitignore` adds `log/`.

It was verified with an end-to-end CPU run of `main()` using a 2-layer model and synthetic shards (data loading, accumulation, all three evals, checkpoint write/reload, and the log format). The DDP path was not exercised, because no GPU was available.

---

## Bugs found along the way

| Introduced | Bug | Effect | Fixed |
|---|---|---|---|
| `d19821f` (Step 4) | Wrap check compared `current_position` to itself + B·T + 1 | Loader would run past the end of the data | `2937a14` (Step 7) |
| `ccf1238` (Step 6) | `config.n_layers` | `AttributeError` at init | `28cf555` |
| `ccf1238` (Step 6) | `hasattr(torch.backends)` | `TypeError` | `28cf555` |
| `bed9b91` (Step 8), exposed by DDP | `autocast(device_type=device)` with `device='cuda:N'` | autocast rejects `'cuda:0'` under torchrun | `b6375a8` (Step 16) |
| `7b9be23` (Step 13) | `device_type == "cuda"` for fused AdamW | Would silently disable fused AdamW under DDP (`cuda:N`) | `269cc7f` (Step 15), the same commit that added DDP |
| `269cc7f` (Step 15) | `require_backward_grad_sync` set after forward | Gradients all-reduced on every micro-step (slower, still correct) | `61c5075` (Step 20) |
| `269cc7f` (Step 15) | `.gitignore`'d an already-tracked `CLAUDE.md` | Ignore entry has no effect | (still present, harmless) |

## What the full run actually used

The completed 19,073-step run (Step 19) was done with the code as of `b6375a8`, **before** Step 20. Compared with the current code:

- Linear biases used PyTorch's default uniform init, not zeros.
- Under DDP, gradients were all-reduced on every micro-step instead of only the last one. That cost speed, not correctness.

Everything else (architecture, data, hyperparameters, evals, seeds, log format) is the same.
