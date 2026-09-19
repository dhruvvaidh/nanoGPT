from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

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
                std *= (2*self.config.n_layers) **-0.5

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



# -------------------------------------------------------
class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T

        # Load the tokens and store them in the memory
        enc = tiktoken.get_encoding('gpt2')
        with open('datasets/tiny_shakespeare/input.txt') as f:
            text = f.read()
        self.tokens = enc.encode(text)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (B*T)} batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B,T = self.B,self.T
        buf = torch.tensor(self.tokens[self.current_position:self.current_position+B*T+1])
        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B,T) # targets
        self.current_position += B*T
        if self.current_position > self.current_position+B*T+1:
            self.current_position = 0
        return x,y
# -------------------------------------------------------

# Auto detects the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends) and torch.backends.mps.is_available():
    device = 'mps'

torch.manual_seed(1337)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)
elif torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# get a databatch
train_loader = DataLoaderLite(B=4,T=32)

num_return_sequences = 5
max_length = 30
#model = GPT.from_pretrained('gpt2')
# We can still run out model, but it'll give us garbage because it hasn't been trained yet
model = GPT(GPTConfig())
model.eval()
model.to(device)

# optimization
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(50):
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    # print(f"Loss: {loss}")
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")


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