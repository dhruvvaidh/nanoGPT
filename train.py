import torch
import time
import math
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from dataloader import DataLoaderLite
from distributed import setup_distributed, cleanup_distributed
from evaluation import estimate_val_loss, evaluate_hellaswag, generate_samples

# ---------------------------------------
# Hyperparameters. There are no CLI flags, edit these in place.
# ---------------------------------------

# We are simulating 0.5 million batch size as mentioned in the paper using gradient accumulation
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size (64 OOMs on 40GB A100s, it needs 80GB)
T = 1024 # sequence length

# All these hyperparameters are derived from GPT3 paper.
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

use_compile = False
# PyTorch compile function increases the performance of the model by reading the code inside the model all at once.
# Advantage 1:
# Normally the python interpreter has to go line by line top read the code but when we're using torch.compile, it tries to analyse the kind of operations we're trying to run.
# Suppose, the ineterpreter will first start reading the forward function to analyse how the opreations are going to be carried out then that code is compiled and
# stored in a seperate object which the python interpreter doesn't reread thus making code execution optimized.
# Advantage 2:
# torch.compile() basically ensures that if one input is being used for calculation, that input is first stored in the HBM and then it's being used further for calculations.
# when torch get's the overview of the code, it can decide how to optimize the communication time between the GPU chip and the HBM. Basically we're using kernel fusion

# how often we run the val/hellaswag/sampling evals, and how often we checkpoint
eval_interval = 250
checkpoint_interval = 5000

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


def main():
    # simple launch
    # python train.py
    # DDP launch for e.g 8GPU
    # torchrun --standalone --nproc_per_node=8 train.py
    ctx = setup_distributed()
    ddp = ctx.ddp
    ddp_rank = ctx.ddp_rank
    ddp_local_rank = ctx.ddp_local_rank
    ddp_world_size = ctx.ddp_world_size
    device = ctx.device
    device_type = ctx.device_type
    master_process = ctx.master_process

    torch.manual_seed(1337)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1337)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    assert total_batch_size % (B * T* ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    # No. of times gradient is getting accumulated (individual batch's gradients are added to one other) and then a single update to update the gradients of the model
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # get a databatch
    train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='train', master_process=master_process)
    val_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='val', master_process=master_process)
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
    if use_compile:
        model = torch.compile(model)
    # at the end of one epoch of the model, DDP synchronises all the processes and adds the gradients to all processes and then carries on back propagation
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    # optimization
    # optimizer = torch.optim.AdamW(model.parameters(),lr=max_lr,betas=(0.9,0.95),eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = max_lr, device_type=device_type, master_process=master_process)
    for step in range(max_steps):
        t0 = time.time()

        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % eval_interval == 0 or last_step:
            val_loss_accum = estimate_val_loss(model, val_loader, device, device_type, ddp)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % checkpoint_interval == 0 or last_step):
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
        if (step % eval_interval == 0 or last_step) and (not use_compile):
            num_correct_norm, num_total, acc_norm = evaluate_hellaswag(model, device, device_type, ddp, ddp_rank, ddp_world_size)
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % eval_interval == 0) or last_step) and (not use_compile):
            generate_samples(model, device, device_type, ddp_rank, num_return_sequences=4, max_length=32)

        model.train()
        optimizer.zero_grad()
        # We are performing gradient accumulation here
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x,y = train_loader.next_batch()
            x,y = x.to(device),y.to(device)
            # We are only performing gradient sync at the final micro step to avoid repeated all-reduces across the number of processes.
            # This flag must be set BEFORE the forward pass: DDP reads it inside forward() to decide whether to arm gradient sync for the backward.
            if ddp:
                model.require_backward_grad_sync = (micro_step==grad_accum_steps-1)
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

    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()


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
