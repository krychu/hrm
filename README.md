Implementation of <a href="https://arxiv.org/abs/2506.21734">Hierachical Reasoning Model</a> (HRM) proposed by Guan Wang et. al.

In a nutshell: the architecture is inspired by the hierarchical and multi-timescale processing in the human brain. In terms of the implementation, there are two connected recurrent modules iterating at different frequencies. The H module iterates slower and represents abstract planning, while L module iterates faster and represents low-level computations. H and L are based on self-attention. The architecture is an approach for carrying reasoning in latent space.

I've done the implementation for educational purposes. I make a few simplifications and extensions to the original work (outlined below). I'd classify these as minor.

I apply the model to a problem of path finding on a board with obstacles: given an NxN board with obstacles, find the shortest path connecting START with END. Example:

<img width="300" src="https://github.com/user-attachments/assets/5bea57e8-5bec-4843-a945-25c49c0c4f1c" />

```
Legend:
.    Floor
#    Wall (obstacle)
S    Start point
E    End point
*    Path (present only in the solution)
```

Input board
```
. . . . S # # . . .
. # . . . # # # # .
. . # # . . # . # #
. . # # . . . # # #
# . . . . # . # . #
# # . . # # # . . #
# . . # . . # . . .
. . . . . . E # . .
. . . . . . . . . .
. . # # . . . . . .
```

Target board (solution)
```
. . . . S # # . . .
. # . . * # # # # .
. . # # * . # . # #
. . # # * . . # # #
# . . * * # . # . #
# # * * # # # . . #
# . * # . . # . . .
. . * * * * E # . .
. . . . . . . . . .
. . # # . . . . . .
```

Results

## Usage

The only dependencies are Python, PyTorch and Pillow: `python -m pip install torch pillow`.

To train a model:
`> python boardpath.py --train`

To use a model on a random board:
`> python boardpath.py --inference`

To tweak the problem, model and training edit: `get_config()` and `get_train_config()` in `boardpath.py`. For example:
- To change the board size and probability of obstacle on the board modify: `board_size` and `wall_prob`.
- To change dimensionality of board cell representation change `d_model`
- To change the number of H and L iterations change `H_cycle_cnt` and `L_cycle_cnt`

All parameters are explained in `hrm/hrm.py`.

## Modifications to the original work

- No Q learning / ACT halting. I simplify by adopting fixed "think time" (that is, fixed # of segments, both at train & inference).
- I use PyTorch's SDPA instead of FlashAttention (same result, less perf).
- I don't use truncated LeCun normal for initialization.
- In addition to RoPE, I added optional learnable absolute positional encoding (`use_abs_pos`) which are added to token embeddings.
- Initial H and L states are learnable (or optionally fixed).
- I use `nn.RMSNorm` with learnable scale (`elementwise_affine=True`).
- There are differences as to where `nn.Dropout` is applied.

## Model architecture outline

- HRM consists of `InputEmbedding`, two instances of `ReasoningModule` (one for H and one for L), and a linear projections. HRM iterates and interconnects H and L modules.
- `InputEmbedding` consists of `nn.Embedding` for input tokens, and optional `nn.Embedding` for absolute token positions (added to token embeddings).
- `ReasoningModule` is a stack of `HRMBlock` modules.
- `HRMBlock` consists of two sublayers:
  1. Attention sublayer: `SDPAttention` -> `nn.Dropout` -> residual connection -> `nn.RMSNorm` (post-norm).
  2. MLP sublayer: `SwiGLU` -> `nn.Dropout` -> residual connection -> `nn.RMSNorm`.
- `SDPAttention` consists of Scaled Dot-Product Attention followed by a linear projection. The module can optionally apply RoPE.

## Findings

per-token average loss across all segments
