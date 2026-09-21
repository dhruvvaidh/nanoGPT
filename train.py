from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time
import math
import inspect
import os
import sys
# `datasets` is also the name of the HuggingFace package (used by fineweb.py), which shadows our local
# datasets/ folder, so put datasets/hellaSwag on sys.path and import the module directly instead.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "hellaSwag"))
from hellaswag import render_example, iterate_examples
#---------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask, registered as a buffer so it moves with model.to(device)
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                   .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention (same as y = q*k.transpose(-2,-1)**(head_size**-0.5))
        # Replacing the scaled_dot_product attention with flash attention (used the same approach for GPT implementation)
        # This is a kernel fusion algo for attention. This is 7.6 times faster because the attention matrix (att) is never actually read/ stored in HBM of the GPU
        # att = (q @ k.transpose(-2,-1)) * (1.0/ math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        ### Later Read why are be creating the MLP in this way? Where is the source for this? Was this there in the og transformer paper?
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        # We are using tanh approximation in GeLU instead of the og implementation because it was slower in tf few years ago
        # Since then using tanh has become the standard practice in pytorch and tf.

        ### Later read why GeLU is being used instead of ReLU for training transformer like models?
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))

# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        # Attention Layer is where the tokens communicate with each other, it's basically a pooling operation or an aggregation function
        # A weighted sum of the different token embeddings in the 1024 token sequence takes place. This is essentially a reduce operation
        x = x+ self.attn(self.ln_1(x))
        # Whereas in MLP, we are applying the operation on every single token there is no information is being collected or exchanged.
        # Essentially Attention is reduce operation and MLP is Map.
        x = x+ self.mlp(self.ln_2(x))

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token (this token delimits documents and can also start generation)
    n_layer:int = 12
    n_head:int = 12
    n_embd:int = 768

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Final Projection - Final Clasifier layer which predicts the tokens
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # Weight sharing scheme (explained in the walkthrough)
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        # the apply function iterates through all the modules in our NN and applies _init_weights function
        self.apply(self._init_weights)


    # The initialization values have been derived from OpenAI's GPT2 code
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # We are scaling down the initialization of the residual layers in the last Linear Transformation layers in the blocks. 
            # We have used the flag NANOGPT_SCALE_INIT  to find the linear transformation layers.
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                # We are getting the below value for the following reasons:
                # 1. In GPT2's paper it is written that the linear transfomations should be scaled down to Number of layers ** -0.5
                # 2. Since we have residual connections going in twice once in MLP and the other in MHA block that is why are taking twice the number of layers in our initialization 
                std *= (2*self.config.n_layer) **-0.5

            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)


    def forward(self,idx,targets = None):
        B,T = idx.shape
        assert T<=self.config.block_size, f"Cannot forward sequence of length {T}, block_size {self.config.block_size}"
        tok_emb = self.transformer.wte(idx) # (B,T,n_embd)
        pos_emb = self.transformer.wpe(torch.arange(0,T,dtype=torch.long,device=idx.device)) # T, n_embd
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        loss = None
        if targets is not None:
            # We need to reshape the multi-dimensional tensor to a single dimension tensor(flattening the tensors) to use Cross Entropy Loss
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # # Create AdamW optimizer and use the fused version if it is available
        # Kernel Fusion happens here as well, instaed of launching a seperate kernel for 1 step of the optimizer, we're doing it all at once in one kernel minimizing overhead.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith("cuda")
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


# -------------------------------------------------------
class DataLoaderLite:
    def __init__(self,B,T,process_rank, num_processes,split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Load the tokens and store them in the memory
        # enc = tiktoken.get_encoding('gpt2')
        # with open('datasets/tiny_shakespeare/input.txt') as f:
        #     text = f.read()
        # self.tokens = enc.encode(text)
        # print(f"Loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens)// (B*T)} batches")

        # get the shard filenames
        data_root = os.path.join("datasets", "fineWeb", "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

        #state
        # if process_rank is 0 then the data will start loading from the 0th position and so on...
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
# -------------------------------------------------------


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


import tiktoken
enc = tiktoken.get_encoding('gpt2')


# simple launch
# python train.py
# DDP launch for e.g 8GPU
# torchrun --standalone --nproc_per_node=8 train.py
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# autocast wants "cuda", not "cuda:0" (which is what device is under torchrun)
device_type = "cuda" if device.startswith("cuda") else device

torch.manual_seed(1337)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)
elif torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# We are simulating 0.5 million batch size as mentioned in the paper using gradient accumulation
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size (64 OOMs on 40GB A100s, it needs 80GB)
T = 1024 # sequence length
assert total_batch_size % (B * T* ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
# No. of times gradient is getting accumulated (individual batch's gradients are added to one other) and then a single update to update the gradients of the model
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# get a databatch
train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='val')
# By default this is set to highest which basically means that even the simplest of matrix multiplications happen on float32 precision (24 mantissa bits 23 stored)
# When we switch it to high we either use tensorfloat 32 (10 mantissa bits stored), consider a f32 as a sum of 2 bloat16 numbers or use faster matrix multi alogs
torch.set_float32_matmul_precision('high')

num_return_sequences = 5
max_length = 30
#model = GPT.from_pretrained('gpt2')
# We have increased the vocab size to 50304 as it's a better number than 50257 as it's divisible with all powers of 2 upto 128 which makes it easier for GPU calculations.
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
use_compile = False
# PyTorch compile function increases the performance of the model by reading the code inside the model all at once.
# Advantage 1:
# Normally the python interpreter has to go line by line top read the code but when we're using torch.compile, it tries to analyse the kind of operations we're trying to run. 
# Suppose, the ineterpreter will first start reading the forward function to analyse how the opreations are going to be carried out then that code is compiled and 
# stored in a seperate object which the python interpreter doesn't reread thus making code execution optimized.
# Advantage 2:
# torch.compile() basically ensures that if one input is being used for calculation, that input is first stored in the HBM and then it's being used further for calculations.
# when torch get's the overview of the code, it can decide how to optimize the communication time between the GPU chip and the HBM. Basically we're using kernel fusion
if use_compile:
    model = torch.compile(model)
# at the end of one epoch of the model, DDP synchronises all the processes and adds the gradients to all processes and then carries on back propagation
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# All these hyperparameters are derived from GPT3 paper.
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
# Recalculating the Learning Rate per step
def get_lr(it):
    #1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr*(it+1)/warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5 * (1.0 + math.cos(math.pi*decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr-min_lr)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# optimization
# optimizer = torch.optim.AdamW(model.parameters(),lr=max_lr,betas=(0.9,0.95),eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = max_lr, device_type=device)
for step in range(max_steps):
    t0 = time.time()

    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    # We are performing gradient accumulation here
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        # We are going to do mixed precision training here
        # Some operations like matrix multiplication in the Linear Layers can be done with lower precision for increased performance.
        # To faciliate certain operations to use automatic mixed precision we use torch.autocast. There is a list of operations which can be autocasted (check the docs)
        # PyTorch documentation recommends us to only use this for training the model and loss calualation and we should leave optimzation and back propagation alone.
        # We don't use flaot16 because if we use them we will have to scale the gradients using Gradient Scaling algos instead we're using bfloat16
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x,y)
        # We are normalizing the loss across multiple micro steps during gradient accumulation. 
        # This is done because we are essentially adding the losses of all the steps and then doing back propagation which increases the gradients therefore we need to normalize them. 
        loss = loss / grad_accum_steps
        # print(f"Loss: {loss}")
        loss_accum += loss.detach()
        # We are only performing gradient accumulation at the final step to avoid repeated accumulations across the number of processes.
        if ddp:
            model.require_backward_grad_sync = (micro_step==grad_accum_steps-1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
    # Calculating the Grad Norm and clipping the global norm to 1.0.
    # During a bad/ unlucky batch if the loss is very high then the gradient which is going to be sent backward can be very high. 
    # This high gradient can shock the model which results in drop in performance. 
    # Therefore, people use this trick to set an upper bound on the norm of the model to prevent this shocking behaviour.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    # This function basically makes the program wait for GPU to finish all it's scheduled tasks
    torch.cuda.synchronize()
    # torch.mps.synchronize() 
    t1 = time.time()
    dt = (t1-t0)*1000 # time difference in milliseconds
    # Toekn processed
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    # Tokens processed per second during training
    tokens_per_sec = tokens_processed/(t1-t0)
    if master_process:
        print(f"step {step:4d}| loss: {loss_accum.item():.6f} | lr: {lr:.4e}| norm: {norm:.2f} | dt: {dt:.2f}ms| tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()


# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens,dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) # Creating 5 copies of the tokens generated above
# x = tokens.to(device)

# generate!

# torch.manual_seed(42)
# torch.mps.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x)
#         # We are only taking logits from the last position (because that'll contain the whole sequence)
#         logits = logits[:,-1,:]
#         # Get the predictive probabilities
#         probs = F.softmax(logits,dim=-1)
#         # Do top k samping of 50 to get the top 50 best choices
#         # We are doing top k sampling as it's the same method used by huggingface's pipelines which is used for generation
#         topk_probs, topk_indices = torch.topk(probs,50,dim=-1)
#         # select a token from the top-k probabilities (multinomial function in torch randomly draws samples from an input tensor of predictive probabilities)
#         ix = torch.multinomial(topk_probs,1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices,-1,ix)
#         # append to the sequence
#         x = torch.cat((x,xcol),dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i,:max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">",decoded)