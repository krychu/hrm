from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import *
import random
import numpy as np

@dataclass
class HRMParameters:
    vocab_cnt: int # Number of classes per sequence item
    seq_len: int # Can be a flattened board

    # attention
    d_model: int # Dim of embeddings and hidden state
    head_cnt: int # Number of heads in attention
    sdpa_dropout: float
    bias_qkv: bool
    bias_o: bool

    # hrm block
    expansion: float # SwiGLU hidden size multiplier (after attn)
    elementwise_affine: bool # rms norm learnable scale
    dropout: float # after attention and mlp

    # reasoning module(s)
    H_block_cnt: int # Number of stacked HRMBlocks in H module
    L_block_cnt: int # Number of stacked HRMBlocks in L module
    H_cycle_cnt: int # Number of H module iterations
    L_cycle_cnt: int # Number of L module iterations per one H iteration

    # hrm
    learnable_z_init: bool # Fixed vs learnable zH_init and zL_init
    infer_segment_cnt: int # Number of forward passes per batch at inference (think time)
    use_rope: bool # Use RoPE (rotary positioning encoding)
    use_abs_pos: bool # Use absolute position encoding

    # head
    head_bias: bool

@dataclass
class HRMTrainParameters:
    train_segment_cnt: int # Number of forward passes per batch, at training time (backprop after each segment)
    epoch_cnt: int
    weight_decay: float
    grad_clip: float | None
    batch_size: int
    lr: float

class InputEmbedding(nn.Module):
    def __init__(
            self,
            vocab_cnt: int,
            seq_len: int,
            embedding_dim: int,
            use_abs_pos: bool
    ):
        super().__init__()
        self.use_abs_pos = use_abs_pos
        self.tok = nn.Embedding(vocab_cnt, embedding_dim)
        if use_abs_pos:
            self.pos = nn.Embedding(seq_len, embedding_dim)
            self.register_buffer("pos_idx", torch.arange(seq_len, dtype=torch.long), persistent=False)
        self.scale = math.sqrt(embedding_dim)

        # optional
        nn.init.normal_(self.tok.weight, mean=0.0, std=1.0/math.sqrt(embedding_dim))
        if use_abs_pos:
            nn.init.normal_(self.pos.weight, mean=0.0, std=1.0/math.sqrt(embedding_dim))

    def forward(self, x_bs: torch.Tensor) -> torch.Tensor:
        """
        x_bs: LongTensor [B,S] (int tokens)
        y_bsd: FloatTensor [B,S,D]

        S = seq_len
        D = embedding_dim
        """
        x_bsd_tok = self.tok(x_bs) # [B, S, D]
        if self.use_abs_pos:
            x_sd_pos = self.pos(self.pos_idx) # [S, D]
            x_1sd_pos = x_sd_pos.unsqueeze(0) # [1, S, D]
            # TODO: no additional scaling by 1/sqrt(2)?
            x_bsd_tok = x_bsd_tok + x_1sd_pos # * (1 / math.sqrt(2))
        y_bsd = self.scale * x_bsd_tok
        return y_bsd

# For RoPE pairs we use concatenated layout, instead of interleaved. For
# (a,b,c,d) the pairs are (a,c) and (b,d).
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., Dh], Dh must be even
    Dh = x.shape[-1]
    x1 = x[..., :Dh // 2]
    x2 = x[..., Dh // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q,k: [B, H, S, Dh]
    cos,sin: [S, Dh]  (broadcasted to [B,H,S,Dh])
    Returns roped (q, k) with original dtypes preserved.
    """
    q_dtype, k_dtype = q.dtype, k.dtype

    # Option A (stable): promote to cos/sin dtype (usually fp32)
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)


    # Option B (fast): keep q/k dtype, cast tables down
    # cos = cos.to(q.dtype)
    # sin = sin.to(q.dtype)

    # Broadcast cos/sin over batch and heads
    cos_ = cos.unsqueeze(0).unsqueeze(0)  # [1,1,S,Dh]
    sin_ = sin.unsqueeze(0).unsqueeze(0)  # [1,1,S,Dh]

    q = (q * cos_) + (rotate_half(q) * sin_)
    k = (k * cos_) + (rotate_half(k) * sin_)
    return q.to(q_dtype), k.to(k_dtype)

class RotaryEmbedding(torch.nn.Module):
    """
    Precomputes cos/sin tables for RoPE.
    - head_dim: per-head dimension (Dh), must be even
    - max_position_embeddings: maximum S you will use
    - base (theta): standard 10000.0
    Buffers move with the module device/dtype via .to / .cuda().
    """
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.max_pos = max_position_embeddings

        # inv_freq: [Dh/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # positions: [S]
        t = torch.arange(self.max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [S, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, Dh]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int | None = None):
        """
        Returns cos,sin of shape [S, Dh] for given S (defaults to max_pos).
        """
        if seq_len is None:
            seq_len = self.max_pos
        if seq_len > self.max_pos:
            raise ValueError(f"Requested RoPE seq_len {seq_len} > max {self.max_pos}")
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class SDPAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            head_cnt: int,
            sdpa_dropout: float,
            bias_qkv: bool,
            bias_o: bool
    ):
        super().__init__()
        self.head_cnt = head_cnt
        self.sdpa_dropout = sdpa_dropout
        self.head_dim = d_model // head_cnt
        assert d_model % head_cnt == 0

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=bias_qkv)
        self.w_o = nn.Linear(d_model, d_model, bias=bias_o)

    def forward(self, x_bsd: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        B, S, D = x_bsd.shape

        qkv_bs3d = self.w_qkv(x_bsd)
        q_bsd, k_bsd, v_bsd = qkv_bs3d.chunk(3, dim=-1)

        # [B,S,D] -> [B,H,S,Dh]
        def split_heads(t_bsd: torch.Tensor) -> torch.Tensor:
            t_bshd = t_bsd.view(B, S, self.head_cnt, self.head_dim)
            t_bhsd = t_bshd.transpose(1, 2) # .continuous()?
            return t_bhsd

        # def split_heads(t_bsd: torch.Tensor) -> torch.Tensor:
        #     return t_bsd.reshape(B, S, self.head_cnt, self.head_dim).transpose(1, 2)

        q_bhsd = split_heads(q_bsd)
        k_bhsd = split_heads(k_bsd)
        v_bhsd = split_heads(v_bsd)

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            q_bhsd, k_bhsd = apply_rope(q_bhsd, k_bhsd, cos, sin)

        x_bhsd = F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd,
            dropout_p=(self.sdpa_dropout if self.training else 0.0)
        ) # [B,H,S,Dh]

        x_bshd = x_bhsd.transpose(1, 2).contiguous()
        x_bsd = x_bshd.view(B, S, D) # [B,S,D]

        x_bsd = self.w_o(x_bsd)
        return x_bsd

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: float):
        super().__init__()
        # ref snaps to multiples of 256; we simplify
        inner = int(expansion * d_model * 2 / 3)
        self.w0 = nn.Linear(d_model, 2 * inner, bias=False) # TODO: check paper for False bias
        self.w1 = nn.Linear(inner, d_model, bias=False) # TODO: check

    def forward(self, x: torch.Tensor):
        g, u = self.w0(x).chunk(2, dim=-1)
        return self.w1(F.silu(g) * u)

class HRMBlock(nn.Module):
    """Attention -> residual -> RMSNorm -> MLP -> residual -> RMSNorm"""
    def __init__(
            self,
            d_model: int,
            head_cnt: int,
            sdpa_dropout: float,
            bias_qkv: bool,
            bias_o: bool,
            expansion: float,
            elementwise_affine: bool,
            dropout: float # after attn and mlp
    ):
        super().__init__()
        self.attn = SDPAttention(
            d_model=d_model,
            head_cnt=head_cnt,
            sdpa_dropout=sdpa_dropout,
            bias_qkv=bias_qkv,
            bias_o=bias_o
        )
        self.mlp = SwiGLU(d_model, expansion)
        self.norm0 = nn.RMSNorm(d_model, elementwise_affine=elementwise_affine)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=elementwise_affine)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_bsd: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor] | None):
        # Attention sublayer
        x_bsd = self.norm0( (x_bsd + self.drop(self.attn(x_bsd, cos_sin=cos_sin))) )
        # MLP sublayer
        x_bsd = self.norm1( (x_bsd + self.drop(self.mlp(x_bsd))) )
        return x_bsd

class ReasoningModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            head_cnt: int,
            sdpa_dropout: float,
            bias_qkv: bool,
            bias_o: bool,
            expansion: float,
            elementwise_affine: bool,
            dropout: float,
            hrm_block_cnt: int,
    ):
        super().__init__()
        self.hrm_blocks = nn.ModuleList([
            HRMBlock(
                d_model=d_model,
                head_cnt=head_cnt,
                sdpa_dropout=sdpa_dropout,
                bias_qkv=bias_qkv,
                bias_o=bias_o,
                expansion=expansion,
                elementwise_affine=elementwise_affine,
                dropout=dropout
            ) for _ in range(hrm_block_cnt)
        ])

    def forward(
            self,
            x_bsd: torch.Tensor,
            x_bsd_injection: torch.Tensor,
            cos_sin: Tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        x_bsd = x_bsd + x_bsd_injection
        for hrm_block in self.hrm_blocks:
            x_bsd = hrm_block(x_bsd, cos_sin=cos_sin)
        return x_bsd

class HRM(nn.Module):
    def __init__(
            self,
            vocab_cnt: int,
            seq_len: int,
            d_model: int,
            head_cnt: int,
            sdpa_dropout: float,
            bias_qkv: bool,
            bias_o: bool,
            expansion: float,
            elementwise_affine: bool,
            dropout: float,
            H_block_cnt: int,
            L_block_cnt: int,
            H_cycle_cnt: int,
            L_cycle_cnt: int,
            learnable_z_init: bool,
            infer_segment_cnt: int,
            use_rope: bool,
            use_abs_pos: bool,
            head_bias: bool
    ):
        super().__init__()
        self.seq_len = seq_len
        self.H_cycle_cnt = H_cycle_cnt
        self.L_cycle_cnt = L_cycle_cnt
        self.embed = InputEmbedding(
            vocab_cnt=vocab_cnt,
            seq_len=seq_len,
            embedding_dim=d_model,
            use_abs_pos=use_abs_pos
        )
        self.rotary = None
        if use_rope:
            assert(d_model // head_cnt) % 2 == 0
            self.rotary = RotaryEmbedding(
                head_dim=d_model // head_cnt,
                max_position_embeddings=seq_len,
                base=10000.0
            )
        self.H = ReasoningModule(
            d_model=d_model,
            head_cnt=head_cnt,
            sdpa_dropout=sdpa_dropout,
            bias_qkv=bias_qkv,
            bias_o=bias_o,
            expansion=expansion,
            elementwise_affine=elementwise_affine,
            dropout=dropout,
            hrm_block_cnt=H_block_cnt
        )
        self.L = ReasoningModule(
            d_model=d_model,
            head_cnt=head_cnt,
            sdpa_dropout=sdpa_dropout,
            bias_qkv=bias_qkv,
            bias_o=bias_o,
            expansion=expansion,
            elementwise_affine=elementwise_affine,
            dropout=dropout,
            hrm_block_cnt=L_block_cnt
        )
        self.head = nn.Linear(
            in_features=d_model,
            out_features=vocab_cnt,
            bias=head_bias
        )

        # Learnable initial templates (broadcast to [B, S, D] on demand)
        if learnable_z_init:
            self.zH_init = nn.Parameter(torch.zeros(1, 1, d_model))
            self.zL_init = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.zH_init, mean=0.0, std=1.0/math.sqrt(d_model))
            nn.init.normal_(self.zL_init, mean=0.0, std=1.0/math.sqrt(d_model))
        else:
            self.register_buffer("zH_init", torch.empty(1,1,d_model).normal_(0, 1/math.sqrt(d_model)))
            self.register_buffer("zL_init", torch.empty(1,1,d_model).normal_(0, 1/math.sqrt(d_model)))


    @torch.no_grad() # Not strictly needed here
    def init_z(
            self,
            x_bs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Broadcast learned templates to [B,S,D]. Call once before the first
        segment."""
        B, S = x_bs.shape
        assert S == self.seq_len
        device = x_bs.device
        dtype = next(self.embed.parameters()).dtype
        zH = self.zH_init.to(device=device, dtype=dtype).expand(B, S, -1).contiguous()
        zL = self.zL_init.to(device=device, dtype=dtype).expand(B, S, -1).contiguous()
        return zH, zL

    def forward(
            self,
            zs_bsd: Tuple[torch.Tensor, torch.Tensor],
            x_bs: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        zs_bsd: (zH, zL) each [B,S,D]
                Hidden state from previous segment (detached before passing in)
        x_bs: [B,S], token ids (ints)
        returns: (zH, zL), logits [B,S,C]

        S = seq_len
        D = d_model
        """

        zH_bsd, zL_bsd = zs_bsd
        x_bsd = self.embed(x_bs)

        # Get RoPE tables once per call (buffers already on correct device)
        S = x_bs.size(1) # seq len
        cos_sin = self.rotary(S) if self.rotary is not None else None # (cos, sin) each [S, Dh]

        with torch.no_grad():
            for idx in range(self.H_cycle_cnt * self.L_cycle_cnt - 1):
                zL_bsd = self.L(zL_bsd, zH_bsd + x_bsd, cos_sin=cos_sin)

                if (idx+1) % self.L_cycle_cnt == 0:
                    zH_bsd = self.H(zH_bsd, zL_bsd, cos_sin=cos_sin)

        zL_bsd = self.L(zL_bsd, zH_bsd + x_bsd, cos_sin=cos_sin)
        zH_bsd = self.H(zH_bsd, zL_bsd, cos_sin=cos_sin)

        y_logits_bsc = self.head(zH_bsd)
        return (zH_bsd, zL_bsd), y_logits_bsc

def train_one_epoch(
        hrm: HRM,
        loader: DataLoader,
        ce_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        segment_cnt: int,
        grad_clip: float | None =None
):
    assert(ce_loss.reduction == "mean")
    hrm.train()

    total_loss = 0.0
    total_tokens = 0

    batch_cnt = len(loader)
    for batch_idx, (x_bs, y_bs) in enumerate(loader):
        print(f"\rbatch: {batch_idx+1}/{batch_cnt}", end='', flush=True)
        x_bs = x_bs.to(device)
        y_bs = y_bs.to(device)
        B, S = x_bs.shape

        # initialize hidden state once per batch
        z_bsd = hrm.init_z(x_bs) # (zH, zL), each [B,S,D]

        for seg_idx in range(segment_cnt):
            optimizer.zero_grad(set_to_none=True)

            z_bsd, logits_bsc = hrm(z_bsd, x_bs)
            # CrossEntropyLoss expects [B,C,S]
            logits_bcs = logits_bsc.transpose(1, 2)
            loss = ce_loss(logits_bcs, y_bs)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(hrm.parameters(), grad_clip)

            optimizer.step()

            # 1-step grad across segments: stop gradients flowing through segments
            zH, zL = z_bsd
            z_bsd = (zH.detach(), zL.detach())

            # With reduction="mean" the loss is an average of per-token loss
            # for this batch/segment. This loss is back propagated.
            #
            # For reporting, we sum the loss across all tokens (*B*S),
            # segments, and batches. We then divide by all tokens that
            # participated.
            #
            # This gives us per-token loss averaged across segments. Exactly
            # what we optimize during training.
            #
            # Alternatively, we could take the last's segment loss.
            total_loss += float(loss.detach()) * B * S
            total_tokens += B * S

    print("\r", end='', flush=True)
    return total_loss / total_tokens

@torch.no_grad()
def evaluate(
    hrm: HRM,
    ce_loss: nn.Module,
    segment_cnt: int,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    For loss reporting: average per-token CE across all segments (same as
    training.

    For accuracy: use only final segment predictions.
    )"""
    hrm.eval()
    assert(ce_loss.reduction == "mean")
    total_loss = 0.0
    total_loss_tokens = 0
    total_correct = 0
    total_tokens = 0
    total_correct_samples = 0
    total_samples = 0

    for x_bs, y_bs in loader:
        x_bs = x_bs.to(device)
        y_bs = y_bs.to(device)
        B, S = x_bs.shape

        # fresh state each eval batch (stateless)
        z = hrm.init_z(x_bs)

        # Run a fixed number of segments (think time)
        for _ in range(segment_cnt):
            z, logits_bsv = hrm(z, x_bs)
            z = (z[0].detach(), z[1].detach())

            loss = ce_loss(logits_bsv.transpose(1, 2), y_bs)
            total_loss += float(loss.detach()) * B * S
            total_loss_tokens += B * S

        preds = logits_bsv.argmax(dim=-1) # [B,S]

        total_correct += (preds == y_bs).sum().item()
        total_tokens += y_bs.numel() # B * S
        total_correct_samples += count_matching_corresponding_rows(preds, y_bs)
        total_samples += preds.size(0)

    avg_loss = total_loss / total_loss_tokens
    acc_cells = total_correct / total_tokens
    acc_samples = total_correct_samples / total_samples
    return avg_loss, acc_cells, acc_samples

def setup_model_and_device(hrm_params: HRMParameters) -> Tuple[HRM, torch.device]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")

    hrm = HRM(
        vocab_cnt=hrm_params.vocab_cnt,
        seq_len=hrm_params.seq_len,
        d_model=hrm_params.d_model,
        head_cnt=hrm_params.head_cnt,
        sdpa_dropout=hrm_params.sdpa_dropout,
        bias_qkv=hrm_params.bias_qkv,
        bias_o=hrm_params.bias_o,
        expansion=hrm_params.expansion,
        elementwise_affine=hrm_params.elementwise_affine,
        dropout=hrm_params.dropout,
        H_block_cnt=hrm_params.H_block_cnt,
        L_block_cnt=hrm_params.L_block_cnt,
        H_cycle_cnt=hrm_params.H_cycle_cnt,
        L_cycle_cnt=hrm_params.L_cycle_cnt,
        learnable_z_init=hrm_params.learnable_z_init,
        infer_segment_cnt=hrm_params.infer_segment_cnt,
        use_rope=hrm_params.use_rope,
        use_abs_pos=hrm_params.use_abs_pos,
        head_bias=hrm_params.head_bias
    ).to(device)

    return hrm, device

def hrm_summary(hrm_params: HRMParameters, hrm_train_params: HRMTrainParameters, hrm: HRM, device: torch.device) -> None:
    trainable_params = sum(p.numel() for p in hrm.parameters() if p.requires_grad)

    print("HRM Parameters:")
    print("-" * 31)
    print(f"{'seq_len':<20} {hrm_params.seq_len:>10}")
    print(f"{'vocab_cnt':<20} {hrm_params.vocab_cnt:>10}")
    print(f"{'d_model':<20} {hrm_params.d_model:>10}")
    print(f"{'head_cnt':<20} {hrm_params.head_cnt:>10}")
    print(f"{'sdpa_dropout':<20} {hrm_params.sdpa_dropout:>10}")
    print(f"{'bias_qkv':<20} {hrm_params.bias_qkv:>10}")
    print(f"{'bias_o':<20} {hrm_params.bias_o:>10}")
    print(f"{'expansion':<20} {hrm_params.expansion:>10}")
    print(f"{'elementwise_affine':<20} {hrm_params.elementwise_affine:>10}")
    print(f"{'dropout':<20} {hrm_params.dropout:>10}")
    print(f"{'H_block_cnt':<20} {hrm_params.H_block_cnt:>10}")
    print(f"{'L_block_cnt':<20} {hrm_params.L_block_cnt:>10}")
    print(f"{'H_cycle_cnt':<20} {hrm_params.H_cycle_cnt:>10}")
    print(f"{'L_cycle_cnt':<20} {hrm_params.L_cycle_cnt:>10}")
    print(f"{'learnable_z_init':<20} {hrm_params.learnable_z_init:>10}")
    print(f"{'infer_segment_cnt':<20} {hrm_params.infer_segment_cnt:>10}")
    print(f"{'use_rope':<20} {hrm_params.use_rope:>10}")
    print(f"{'use_abs_pos':<20} {hrm_params.use_abs_pos:>10}")
    print(f"{'head_bias':<20} {hrm_params.head_bias:>10}")

    print("\nHRM Training Parameters:")
    print("-" * 31)
    print(f"{'train_segment_cnt':<20} {hrm_train_params.train_segment_cnt:>10}")
    print(f"{'epoch_cnt':<20} {hrm_train_params.epoch_cnt:>10}")
    print(f"{'weight_decay':<20} {hrm_train_params.weight_decay:>10}")
    grad_clip_str = "None" if hrm_train_params.grad_clip is None else str(hrm_train_params.grad_clip)
    print(f"{'grad_clip':<20} {grad_clip_str:>10}")
    print(f"{'batch_size':<20} {hrm_train_params.batch_size:>10}")
    print(f"{'lr':<20} {hrm_train_params.lr:>10}")

    print("\nModel Statistics:")
    print("-" * 31)
    print(f"{'trainable_params':<20} {trainable_params:>10_}")
    print(f"{'device':<20} {str(device):>10}")
    print()

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def count_matching_corresponding_rows(a: torch.Tensor, b: torch.Tensor) -> int:
    assert(len(a.shape)==2 and len(b.shape)==2)
    assert(a.shape == b.shape)
    matches = (a == b).all(dim=1)
    return int(matches.sum().item())
