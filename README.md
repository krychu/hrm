## Hierarchical Reasoning Model (HRM)

This is an implementation of the <a href="https://arxiv.org/abs/2506.21734">Hierachical Reasoning Model</a> (HRM) proposed by Guan Wang et al. I built it for educational purposes, with a few minor simplifications and extensions (see [Modifications to the Original Work](#modifications-to-the-original-work)).

The architecture is inspired by hierarchical, multi-timescale processing in the human brain. It uses two recurrent modules running at different frequencies:
- **H module** (slower): handles abstract planning
- **L module** (faster): handles low-level computations

Both are based on self-attention. Together, they enable reasoning in latent space.

## Demo application

The model is applied to a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END. The animation below shows actual inference steps as the model incrementally discovers the path.

<img width="300" src="https://github.com/user-attachments/assets/5bea57e8-5bec-4843-a945-25c49c0c4f1c" />

Legend: . = Floor, \# = Wall, S = Start point, E = End point, * = Path

## Usage

Dependencies: Python, PyTorch and Pillow <br/>
```bash
python -m pip install torch pillow
```

Train a model: <br/>
```bash
python boardpath.py --train
```

Run inference on a random board (also saves an animated GIF of the steps): <br/>
```
python boardpath.py --inference
```

To adjust the task, model, or training setup, edit `get_config()` and `get_train_config()` in `boardpath.py`. For example:
- Board size & obstacle density: `board_size`, `wall_prob`
- Embedding dimensionality (representation of each board cell): `d_model`
- \# of iterations ("think time"): `H_cycle_cnt`, `L_cycle_cnt`

All parameters are documented in `hrm/hrm.py`.

## Modifications to the Original Work

- Fixed "think time" (that is, fixed # of segments, both in train & inference) instead of Q-learning / ACT halting
- PyTorch SDPA instead of FlashAttention (same results, lower performance)
- Standard initialization instead of truncated LeCun normal
- Optional learnable absolute positional encoding (`use_abs_pos`) in addition to RoPE
- Learnable (or fixed) initial H and L states
- `nn.RMSNorm` with learnable scale (`elementwise_affine=True`)
- Slight differences in where `nn.Dropout` is applied

## Model architecture

- `HRM` = `InputEmbedding` + two `ReasoningModule` instances (H and L) + linear projection (on last H state)
- `InputEmbedding` = `nn.Embedding` for tokens + optional absolute positional `nn.Embedding` (added to token embeddings)
- `ReasoningModule` = stack of `HRMBlock` instances
- `HRMBlock` has two sublayers:
  1. Attention: `SDPAttention` → `Dropout` → residual → `RMSNorm` (post-norm)
  2. MLP: `SwiGLU` → `Dropout` → residual → `RMSNorm`
- `SDPAttention` = scaled dot-product attention + linear projection, with optional RoPE
