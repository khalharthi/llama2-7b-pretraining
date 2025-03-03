import torch
import torch.nn as nn
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
#from transformers import AutoTokenizer

torch.cuda.empty_cache() # Free up the GPU memory held by the caching allocator before starting training
@dataclass
class LLAMAConfig:
    block_size: int = 4096
    vocab_size: int = 200000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


class ApplyRotaryEmbedings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd // config.n_head
        self.max_seq_len = config.block_size
        self.base = 10000.0

        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float()/self.dim)) # Shape: (dim // 2, )
        freqs = torch.outer(torch.arange(self.max_seq_len), theta) #(seq_len, dim//2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) #(seq_len, dim//2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, q, k):
        B, H, T, D = q.shape  # Extract batch size, num_heads, seq_len, head_dim
        assert D % 2 == 0, "Head dimension must be even for complex number conversion."

        # Slice `freqs_cis` to match `T`
        freqs_cis = self.freqs_cis[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)

        # Reshape `q` and `k` to treat the last dimension as complex numbers (real and imaginary parts)
        q_complex = torch.view_as_complex(q.float().reshape(B, H, T, D // 2, 2)) # (B, H, T, D//2, 2) >--(view_as_complex)--> (B, H, T, D//2) note: D here is head_dim
        k_complex = torch.view_as_complex(k.float().reshape(B, H, T, D // 2, 2)) # (B, H, T, D//2, 2) >--(view_as_complex)--> (B, H, T, D//2) note: D here is head_dim

        # Apply RoPE rotation in complex space, and then convert the last dim as real numbers
        q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2, -1) # >--(B, H, T, D//2)--(view_as_real)--(B, H, T, D//2, 2)--(flatten)-->(B, H, T, D) note: D here is head_dim
        k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2, -1) # >--(B, H, T, D//2)--(view_as_real)--(B, H, T, D//2, 2)--(flatten)-->(B, H, T, D) note: D here is head_dim

        return q_out.type_as(q), k_out.type_as(k)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_att = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1 # Attribute Injection, for custom weight initialization
        self.apply_rotary = ApplyRotaryEmbedings(config)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, kv_cache = None):
        B, T, C = x.size()
        qkv = self.c_att(x)
        Q, K, V = qkv.split(self.n_embd, dim=2)
        
        head_dim = self.n_embd // self.n_head
        Q, K, V = [t.view(B, T, self.n_head, head_dim).permute(0, 2, 1, 3) for t in (Q, K, V)] # Q, K, V shapes: (B, n_head, T, head_dim)
        Q,  K = self.apply_rotary(Q, K)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            K = torch.cat([prev_k, K], dim=1)
            V = torch.cat([prev_v, V], dim=1)
            new_kv_cache = (K, V)
        else:
            new_kv_cache = None

        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        expansion_factor = (4 * config.n_embd) / 3 # Expansion factor for SwiGLU
        # Expansion layer (SwiGLU requires 2/3 * expansion factor)
        self.w1 = nn.Linear(config.n_embd, int(2 * expansion_factor)) # Expansion layer
        self.w2 = nn.Linear(int(expansion_factor), config.n_embd)  # Projection layer
        self.w2.SCALE_INIT = 1

    def forward(self, x):
        w1_out = self.w1(x)  # Shape: (B, T, 2/3 * expansion)
        # Split into gate and value 
        gate, value = w1_out.chunk(2, dim=-1)  #Each of shape: (B, T, 1/3 * expansion)
        swiglu_out = F.silu(gate) * value  # SwiGLU activation
        return self.w2(swiglu_out)  # Project back to original dimensionality
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.att = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x, kv_cache=None):
        att_output, kv_cache = self.att(self.ln_1(x), kv_cache=kv_cache) 
        x = x + att_output
        x = x + self.ff(self.ln_2(x))
        return x, kv_cache
    
class LLAMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #weight sharing schema
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module, std=0.02):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'SCALE_INIT'):
                std *= (self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.size() # idx is of shape (B, T)
        assert T <= self.config.block_size, "sequence length {T} is more than block_size"
        
        tok_emb = self.transformer.wte(idx) #(B, T, n_embd)
        x = tok_emb
        
        if kv_cache is None:
            kv_cache = [None] * len(self.transformer.h)

        new_kv_cache = []

        for i, block in enumerate(self.transformer.h):
            x, kv_cache_i = block(x, kv_cache=kv_cache[i])
            new_kv_cache.append(kv_cache_i)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_cache
    
    def configure_optimizer(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_param = [p for n, p in param_dict.items() if "bias" not in n and "norm" not in n]
        nodecay_param = [p for n, p in param_dict.items() if "bias" in n or "norm" in n]

        optim_groups = [
            {'params': decay_param, 'weight_decay': weight_decay},
            {'params': nodecay_param, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_param)
        num_nodecay_params = sum(p.numel() for p in nodecay_param)
        print(f"num decayed parameters tensor: {len(decay_param)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameters tensor: {len(nodecay_param)}, with {num_nodecay_params} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, **extra_args)
        return optimizer

def load_tokens(filename):
    npt= np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:

    def __init__(self, B, T,process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "/home/khaled/ai/gpt/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for the split {split}")
        

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # Position of this process

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = B * T * self.process_rank
        return x, y
    
def get_most_likely_row(tokens, mask, logits): #tokens shape: [B,T], mask shape: [B,T], logits shape: [B, T, V]
    shift_logits = (logits[..., :-1, :]).contiguous() #Removes the last timestep from logits to align it with the next-token targets in shift_tokens. size[B, T-1, V]
    shift_tokens = (tokens[..., 1:]).contiguous() # size[B, T-1]
    flat_shif_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shif_tokens = shift_tokens.view(-1)
    shif_losses = F.cross_entropy(flat_shif_logits, flat_shif_tokens, reduction='none') #size: [20]
    shif_losses = shif_losses.view(tokens.size(0), -1) # [20] --> [B, T-1]
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shif_losses * shift_mask #size: [B,T−1]
    sum_losses = masked_shift_losses.sum(dim=1) #size: [B] 
    avg_loss = sum_losses / shift_mask.sum(dim=1) #size: [B]
    pred_norm = avg_loss.argmin().item()
    return pred_norm

ddp = int(os.environ.get('Rank', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "No Cuda"
    #Step1: initializes the process group, sets up communication between all processes in the group.
    init_process_group(backend='nccl')
    #Step2: retrieves the global rank of the current process from the environment variable RANK. Identifies the role of each process.
    ddp_rank = int(os.environ['RANK'])
    #Step3: retrieves the local rank of the process on the current node (machine) from the environment variable LOCAL_RANK
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    #step4: retrieves the total number of processes participating in the training job from the environment variable WORLD_SIZE.
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda: {ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size =1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"using device: {device}")

enc = tiktoken.get_encoding("o200k_base")


B = 64
T = 4096
total_batch_size = B * T * ddp_world_size # = 2083328 if we use 8 gpus. Each gpu takes 260416 tokens
assert total_batch_size %(B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumlation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')

model = LLAMA(LLAMAConfig(vocab_size=200192))
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 4700

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0<= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device_type=device)
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, 'w') as f:
    pass

for step in range(max_steps):

    t0 = time.time()
    last_step = (step == max_steps -1)

    if step % 250 == 0 or last_step: # Calculate validation loss every 250 step
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss, kv_cache = model(x, y, kv_cache=None)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum:.4f}")
            
            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                check_point_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, check_point_path)

    if (step % 250 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples('val')):
            #only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            #render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step}: hella {acc_norm:4f}\n")

    if ((step > 0 and step % 250 == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 40
        tokens = enc.encode("Hello I'm a language model, ")
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        kv_cache = None
        xgen = tokens
        first_step = True # Flag to track the first step

        # Generate tokens until max_length is reached
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    # First step: Pass full sequence
                    if first_step:
                        logits, _, kv_cache = model(xgen, kv_cache=kv_cache) # logits: (B, T, vocab_size)
                        first_step = False
                    # Subsequent steps: Only pass the last generated token
                    else:
                        logits, _, kv_cache = model(xgen[:, -1:], kv_cache=kv_cache) # logits: (B, 1, vocab_size)
                        
                # Get the logits for the last token
                logits = logits[:, -1, :] #(B, vocab_size)
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample from the top-k tokens
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # topk_probs: (B, 50), topk_indices: (B, 50)
                
                ix = torch.multinomial(topk_probs, 1) # ix: (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # xcol: (B, 1)
                
                # Append the sampled token to the generated sequence
                xgen = torch.cat((xgen, xcol), dim=1) # xgen: (B, T + 1)
                
        # Decode and print the generated sequences        
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss, kv_cache = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4f} | norm: {norm:.4f} | dt: {dt*1000:.2f} ms | tok/sec: {tokens_per_second:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
if ddp:
    destroy_process_group()
