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
    expansion: int # swiglu hidden size multiplier (after attn)
    elementwise_affine: bool # rms norm learnable scale
    dropout: float # after attention and mlp

    # reasoning module(s)
    H_block_cnt: int # Number of stacked HRMBlocks in H module
    L_block_cnt: int # Number of stacked HRMBlocks in L module
    H_cycle_cnt: int # Number of H module iterations
    L_cycle_cnt: int # Number of L module iterations per one H iteration

    # hrm
    infer_segment_cnt: int # Number of forward passes per batch at inference (think time)

    # head
    head_bias: bool

# @dataclass
# class HRMParameters:
#     seq_len: int # Can be a flattened board
#     vocab_cnt: int # Number of classes per sequence item
#     d_model: int # Dim of embeddings and hidden state
#     nhead: int # Number of heads in mha
#     dim_feedforward: int # Dim of ff network in mha
#     H_layer_cnt: int # number of ???
#     L_layer_cnt: int
#     H_cycle_cnt: int # Number of H module iterations
#     L_cycle_cnt: int # Number of L module iterations per one H iteration
#     infer_segment_cnt: int # Number of forward passes per batch at inference time (how hard to think?)
#     dropout: float

@dataclass
class HRMTrainParameters:
    train_segment_cnt: int # Number of forward passes per batch, at training time (backprop after each segment)
    epoch_cnt: int
    weight_decay: float
    grad_clip: float | None
    batch_size: int
    lr: float

# TODO: parametrize, tok: bool, pos: bool
class InputEmbedding(nn.Module):
    def __init__(self, vocab_cnt: int, seq_len: int, embedding_dim: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_cnt, embedding_dim)
        self.pos = nn.Embedding(seq_len, embedding_dim)
        self.scale = math.sqrt(embedding_dim)

        # optional
        nn.init.normal_(self.tok.weight, mean=0.0, std=1.0/math.sqrt(embedding_dim))
        nn.init.normal_(self.pos.weight, mean=0.0, std=1.0/math.sqrt(embedding_dim))

        self.register_buffer("pos_idx", torch.arange(seq_len, dtype=torch.long), persistent=False)

    def forward(self, x_bs: torch.Tensor) -> torch.Tensor:
        """
        x_bs: LongTensor [B,S] (int tokens)
        y_bsd: FloatTensor [B,S,D]

        S = seq_len
        D = embedding_dim
        """
        x_bsd_tok = self.tok(x_bs) # [B, S, D]
        x_sd_pos = self.pos(self.pos_idx) # [S, D]
        x_1sd_pos = x_sd_pos.unsqueeze(0) # [1, S, D]
        y_bsd = self.scale * (x_bsd_tok + x_1sd_pos)
        return y_bsd

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

    def forward(self, x_bsd: torch.Tensor) -> torch.Tensor:
        B, S, D = x_bsd.shape

        qkv_bs3d = self.w_qkv(x_bsd)
        q_bsd, k_bsd, v_bsd = qkv_bs3d.chunk(3, dim=-1)

        def split_heads(t_bsd: torch.Tensor) -> torch.Tensor:
            t_bshd = t_bsd.view(B, S, self.head_cnt, self.head_dim)
            t_bhsd = t_bshd.transpose(1, 2) # .continuous()?
            return t_bhsd

        q_bhsd = split_heads(q_bsd)
        k_bhsd = split_heads(k_bsd)
        v_bhsd = split_heads(v_bsd)

        x_bhsd = F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd,
            dropout_p=(self.sdpa_dropout if self.training else 0.0)
        )

        x_bshd = x_bhsd.transpose(1, 2).contiguous()
        x_bsd = x_bshd.view(B, S, D)

        x_bsd = self.w_o(x_bsd)
        return x_bsd

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: int):
        super().__init__()
        # ref snaps to multiples of 256; we simplify
        inner = int(expansion * d_model * 2 / 3)
        self.w0 = nn.Linear(d_model, 2 * inner, bias=False) # TODO: check
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
            expansion: int,
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
        self.norm0 = nn.RMSNorm(d_model, elementwise_affine)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine)
        # TODO: check that dropout gets reset in eval
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_bsd: torch.Tensor):
        # Attention sublayer
        x_bsd = self.norm0( (x_bsd + self.drop(self.attn(x_bsd))) )
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
            expansion: int,
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

    def forward(self, x_bsd: torch.Tensor, x_bsd_injection: torch.Tensor) -> torch.Tensor:
        x_bsd = x_bsd + x_bsd_injection
        for hrm_block in self.hrm_blocks:
            x_bsd = hrm_block(x_bsd)
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
            expansion: int,
            elementwise_affine: bool,
            dropout: float,
            H_block_cnt: int,
            L_block_cnt: int,
            H_cycle_cnt: int,
            L_cycle_cnt: int,
            infer_segment_cnt: int,
            head_bias: bool
    ):
        super().__init__()
        self.seq_len = seq_len
        # self.d_model = d_model
        self.H_cycle_cnt = H_cycle_cnt
        self.L_cycle_cnt = L_cycle_cnt
        self.embed = InputEmbedding(
            vocab_cnt=vocab_cnt,
            seq_len=seq_len,
            embedding_dim=d_model
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
        self.zH_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.zL_init = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.zH_init, mean=0.0, std=1.0/math.sqrt(d_model))
        nn.init.normal_(self.zL_init, mean=0.0, std=1.0/math.sqrt(d_model))

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

        with torch.no_grad():
            for idx in range(self.H_cycle_cnt * self.L_cycle_cnt - 1):
                zL_bsd = self.L(zL_bsd, zH_bsd + x_bsd)

                if (idx+1) % self.L_cycle_cnt == 0:
                    zH_bsd = self.H(zH_bsd, zL_bsd)

        zL_bsd = self.L(zL_bsd, zH_bsd + x_bsd)
        zH_bsd = self.H(zH_bsd, zL_bsd)

        y_logits_bsc = self.head(zH_bsd)
        return (zH_bsd, zL_bsd), y_logits_bsc

# class ReasoningModule(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                  nhead: int,
#                  dim_feedforward: int,
#                  num_layers: int,
#                  dropout: float = 0.0
#                  ):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True
#         )
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers,
#             enable_nested_tensor=False # suppress test warnings
#         )

#     def forward(
#             self,
#             z_bsd: torch.Tensor,
#             x_bsd_injection: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         z_bsd: FloatTensor [B,S,D]
#         x_bsd_injection: FloatTensor [B,S,D]
#         y_bsd: FloatTensor [B,S,D]

#         D = d_model
#         """
#         x_bsd = z_bsd + x_bsd_injection
#         y_bsd = self.encoder(x_bsd)
#         return y_bsd

# class HRM(nn.Module):
#     def __init__(self,
#                  vocab_cnt: int,
#                  seq_len: int,
#                  d_model: int,
#                  nhead: int,
#                  dim_feedforward: int,
#                  H_layers: int,
#                  L_layers: int,
#                  H_cycles: int,
#                  L_cycles: int,
#                  dropout: float = 0.0,
#                  ):
#         super().__init__()
#         self.seq_len = seq_len
#         self.d_model = d_model
#         self.L_cycles = L_cycles
#         self.H_cycles = H_cycles
#         self.embed = InputEmbedding(
#             vocab_cnt=vocab_cnt,
#             seq_len=seq_len,
#             embedding_dim=d_model
#         )
#         self.H = ReasoningModule(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             num_layers=H_layers,
#             dropout=dropout
#         )
#         self.L = ReasoningModule(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             num_layers=L_layers,
#             dropout=dropout
#         )
#         self.head = nn.Linear(
#             in_features=d_model,
#             out_features=vocab_cnt,
#             bias=False
#         )

#         # Learnable initial templates (broadcast to [B, S, D] on demand)
#         self.zH_init = nn.Parameter(torch.zeros(1, 1, d_model))
#         self.zL_init = nn.Parameter(torch.zeros(1, 1, d_model))
#         nn.init.normal_(self.zH_init, mean=0.0, std=1.0/math.sqrt(d_model))
#         nn.init.normal_(self.zL_init, mean=0.0, std=1.0/math.sqrt(d_model))

#     @torch.no_grad() # Not strictly needed here
#     def init_z(
#             self,
#             x_bs: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Broadcast learned templates to [B,S,D]. Call once before the first
#         segment."""
#         B, S = x_bs.shape
#         assert S == self.seq_len
#         device = x_bs.device
#         dtype = next(self.embed.parameters()).dtype
#         zH = self.zH_init.to(device=device, dtype=dtype).expand(B, S, -1).contiguous()
#         zL = self.zL_init.to(device=device, dtype=dtype).expand(B, S, -1).contiguous()
#         return zH, zL

#     def forward(self,
#                 zs_bsd: Tuple[torch.Tensor, torch.Tensor],
#                 x_bs: torch.Tensor
#                 ):
#         """
#         zs_bsd: (zH, zL) each [B,S,D]
#                 Hidden state from previous segment (detached before passing in)
#         x_bs: [B,S], token ids (ints)
#         returns: (zH, zL), logits [B,S,C]

#         S = seq_len
#         D = d_model
#         """

#         zH_bsd, zL_bsd = zs_bsd
#         x_bsd = self.embed(x_bs)

#         with torch.no_grad():
#             for idx in range(self.H_cycles * self.L_cycles - 1):
#                 zL_bsd = self.L(zL_bsd, zH_bsd + x_bsd)

#                 if (idx+1) % self.L_cycles == 0:
#                     zH_bsd = self.H(zH_bsd, zL_bsd)

#         zL_bsd = self.L(zL_bsd, zH_bsd + x_bsd)
#         zH_bsd = self.H(zH_bsd, zL_bsd)

#         y_logits_bsc = self.head(zH_bsd)
#         return (zH_bsd, zL_bsd), y_logits_bsc

def train_one_epoch(
        hrm: HRM,
        loader: DataLoader,
        ce_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        segment_cnt: int,
        grad_clip: float | None =None
):
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

            total_loss += float(loss.detach()) * B * S
            total_tokens += B * S

    print("\r", end='', flush=True)
    return total_loss / total_tokens

@torch.no_grad()
def evaluate(
    hrm: HRM,
    segment_cnt: int,
    loader: DataLoader,
    device: torch.device,
):
    hrm.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x_bs, y_bs in loader:
        x_bs = x_bs.to(device)
        y_bs = y_bs.to(device)

        # fresh state each eval batch (stateless)
        z = hrm.init_z(x_bs)

        # Run a fixed number of segments (think time)
        for _ in range(segment_cnt):
            z, logits_bsv = hrm(z, x_bs)
            z = (z[0].detach(), z[1].detach())

        loss = ce(logits_bsv.transpose(1, 2), y_bs)
        preds = logits_bsv.argmax(dim=-1) # [B,S]
        total_correct += (preds == y_bs).sum().item()
        total_tokens += y_bs.numel()
        total_loss += float(loss)

    avg_loss = total_loss / total_tokens
    acc = total_correct / total_tokens
    return avg_loss, acc

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
    print(f"{'infer_segment_cnt':<20} {hrm_params.infer_segment_cnt:>10}")
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
