import os, math, time, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Config:
    vocab_size = 256
    emb_dim = 128
    n_heads = 8
    n_layers = 4
    context_len = 256
    ff_dim = emb_dim * 4
    dropout = 0.1

    batch_size = 64
    lr = 3e-4
    weight_decay = 0.01
    max_steps = 30_000
    warmup_steps = 200
    eval_interval = 500
    eval_steps = 100
    grad_clip = 1.0

    data_path = "input.txt"
    out_dir   = "."

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

cfg = Config()

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.emb_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.qkv  = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=False)
        self.proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_len, cfg.context_len))
                .view(1, 1, cfg.context_len, cfg.context_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.ff_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, cfg.emb_dim, bias=False),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.emb_dim)
        self.ff   = FeedForward(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.emb_dim)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.emb_dim)
        self.head    = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx


@torch.no_grad()
def generate_with_penalty(mdl, idx, max_new_tokens, temperature=0.4, top_k=10, repeat_penalty=1.3):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.context_len:]
        logits, _ = mdl(idx_cond)
        logits = logits[:, -1, :] / temperature
        for token_id in set(idx_cond[0].tolist()):
            logits[0, token_id] /= repeat_penalty
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
    return idx

class ByteDataset(Dataset):
    def __init__(self, data: bytes, context_len: int):
        self.data = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.ctx  = context_len
    def __len__(self):
        return len(self.data) - self.ctx
    def __getitem__(self, i):
        return self.data[i:i+self.ctx], self.data[i+1:i+self.ctx+1]

def load_data(path, split=0.9):
    raw = open(path, "rb").read()
    n = int(len(raw) * split)
    return ByteDataset(raw[:n], cfg.context_len), ByteDataset(raw[n:], cfg.context_len)

def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t))

@torch.no_grad()
def estimate_loss(model, val_loader):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_steps: break
        _, loss = model(x.to(cfg.device), y.to(cfg.device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def quantize_tensor(t):
    scale = t.abs().max().item() / 127.0
    if scale == 0: scale = 1e-8
    q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
    return q.cpu().numpy(), scale

def export_weights_int8(model, out_path):
    tensors = {name: p.detach().float() for name, p in model.named_parameters()}
    with open(out_path, "wb") as f:
        f.write(b"TFPGA001")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            q, scale = quantize_tensor(t)
            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", q.ndim))
            for s in q.shape:
                f.write(struct.pack("<I", s))
            f.write(struct.pack("<f", scale))
            f.write(q.tobytes())
    total = sum(np.prod(p.shape) for p in model.parameters())
    print(f"Exported {len(tensors)} tensors -> {out_path}")
    print(f"INT8 weight file: {total / 1024:.1f} KB")

def load_weights_int8(path):
    tensors = {}
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"TFPGA001", "Invalid file"
        n_tensors = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_tensors):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
            scale = struct.unpack("<f", f.read(4))[0]
            data = np.frombuffer(f.read(int(np.prod(shape))), dtype=np.int8)
            tensors[name] = torch.tensor(data.astype(np.float32) * scale).reshape(shape)
    return tensors

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"Device: {cfg.device}")

    train_ds, val_ds = load_data(cfg.data_path)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train tokens: {len(train_ds):,}  |  Val tokens: {len(val_ds):,}")

    model = TinyTransformer(cfg).to(cfg.device)
    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    os.makedirs(cfg.out_dir, exist_ok=True)

    model.train()
    loader_iter = iter(train_loader)
    t0 = time.time()

    for step in range(cfg.max_steps):
        lr = get_lr(step)
        for g in optimizer.param_groups: g["lr"] = lr

        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(cfg.device), y.to(cfg.device)
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            dt = (time.time() - t0) * 1000 / max(step % 100, 1) if step > 0 else 0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}ms/step")
            t0 = time.time()

        if step % cfg.eval_interval == 0 and step > 0:
            vl = estimate_loss(model, val_loader)
            print(f"  >>> val_loss {vl:.4f}")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step, "val_loss": vl,
            }, os.path.join(cfg.out_dir, "ckpt.pt"))

    print("Training complete")

    bin_path = os.path.join(cfg.out_dir, "weights_int8.bin")
    export_weights_int8(model, bin_path)
