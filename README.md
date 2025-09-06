## Hierarchical Reasoning Model (HRM)

This is an implementation of the <a href="https://arxiv.org/abs/2506.21734">Hierachical Reasoning Model</a> (HRM) proposed by Guan Wang et al. I built it for educational purposes, with a few minor simplifications and extensions (see [Modifications to the Original Work](#modifications-to-the-original-work)).

The architecture is inspired by hierarchical, multi-timescale processing in the human brain. It uses two connected recurrent modules running at different frequencies:
- **H module** (slower): handles abstract planning
- **L module** (faster): handles low-level computations

Both are based on self-attention. Together, this is an attempt to model reasoning in latent space.

---

👉 See [**Ablation Study: What Drives HRM Performance?**](#ablation-study-what-drives-performance)

## Demo application: pathfinding

The model is applied to a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END. The animation below shows actual inference steps as the model incrementally discovers the path.

| 10x10 board | 20x20 board |
|-------------|-------------|
| <img width="320" src="https://github.com/user-attachments/assets/5bea57e8-5bec-4843-a945-25c49c0c4f1c" /> | <img width="350" src="https://github.com/user-attachments/assets/26a9a202-b23c-4c1e-ad73-04f858fba8de" />


Legend: . = Floor, \# = Wall, S = Start point, E = End point, * = Path

## Usage

Dependencies: <br/>
```bash
python3 -m pip install torch pillow
```

Train a model: <br/>
```bash
python3 boardpath.py --mode train
```

Run inference on a random board (also saves an animated GIF of the steps): <br/>
```
python3 boardpath.py --mode inference
```

To adjust the task, model, or training setup, edit `get_config()` and `get_train_config()` in `boardpath.py`. For example:
- Board size & obstacle density: `board_size`, `wall_prob`
- Embedding dimensionality (representation of each board cell): `d_model`

All parameters are documented in `hrm/hrm.py`.

## Modifications to the Original Work

- Fixed "think time" (that is, fixed # of segments, both in train & inference) instead of Q-learning / ACT halting
- PyTorch SDPA instead of FlashAttention (same results, lower performance)
- RoPE and learned positional encodings can be used together
- Standard initialization instead of truncated LeCun normal
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

## Ablation Study: What Drives HRM Performance?

Recent discussion around the Hierarchical Reasoning Model (HRM) asks whether it's the new two-timescale H/L architecture that drives the performance.

To explore this, I ran a simple set of ablations on a **board pathfinding task** (20×20 boards, wall probability = 0.3). Each variant was trained on **2000 training boards** and validated on **500 boards**, for **40 epochs** with **lr = 3e-4** and **batch size = 64**. Models were parameter-matched: H/L variants used *d_model = 256* (~6.29M parameters), single-module variants used *d_model = 360* (~6.23M parameters).

This is a small study on a relatively simple task, so the results should be taken as illustrative rather than definitive.

---

### What Was Varied
- **Architecture (variant):**
  - *H-only (detached):* single ReasoningModule unrolled for H×L steps, hidden state detached.
  - *H-only (bptt):* same as above, but with full backpropagation-through-time (BPTT).
  - *H/L:* full HRM with separate H and L modules.
- **Training segments (train_seg):** number of refinement segments during training.
- **Inference segments (infer_seg):** number of refinement segments during evaluation.
- **Cycles:** number of inner iterations (H×L).

---

### Results (all runs)

In the table:
- **board acc** = accuracy of predicting the entire board correctly (last 5 epochs average).
- **acc4x** = the same metric, but with 4× more inference segments (extra test-time refinement).
- **Gap** = acc4x – board acc, showing how much accuracy improves with additional inference steps.

| Variant            | train_seg | infer_seg | H×L cycles | Board acc (last-5) | acc4x (last-5) | Gap   |
|--------------------|-----------|-----------|------------|---------------------|----------------|-------|
| H/L                | 2         | 2         | 2×2 (4)    | 0.390               | 0.382          | -0.008|
| H/L                | 4         | 2         | 2×2 (4)    | 0.320               | 0.425          | +0.104|
| H/L                | 2         | 2         | 2×4 (8)    | 0.403               | 0.444          | +0.041|
| H/L                | 4         | 2         | 2×4 (8)    | 0.447               | 0.552          | +0.105|
| H/L                | 2         | 2         | 4×2 (8)    | 0.458               | 0.481          | +0.023|
| H/L                | 4         | 2         | 4×2 (8)    | 0.523               | 0.545          | +0.022|
| H-only (bptt)      | 2         | 2         | 4          | 0.226               | 0.272          | +0.046|
| H-only (detached)  | 2         | 2         | 4          | 0.158               | 0.182          | +0.024|
| H-only (bptt)      | 4         | 2         | 4          | **0.574**           | **0.625**      | +0.052|
| H-only (detached)  | 4         | 2         | 4          | 0.347               | 0.436          | +0.089|
| H-only (bptt)      | 2         | 2         | 8          | 0.376               | 0.394          | +0.018|
| H-only (detached)  | 2         | 2         | 8          | 0.222               | 0.226          | +0.005|

---

### Charts

**Board accuracy (last 5 epochs average):**

| acc by variant | acc by train_seg | acc by cycles |
|----------------|------------------|---------------|
| img | img | img

**Refinement gap (last 5 epochs average):**

| gap by variant | gap by train_set | gap by cycles |
|----------------|------------------|---------------|
| img | img | img |

---

### Conclusions
1. **Segments are the main driver.**
   They improve both accuracy and refinement ability.

2. **Architecture has little influence.**
   H/L and single-module BPTT perform similarly; any differences are minor compared to the impact of segments.

3. **Cycles increase accuracy but not refinement.**
   More cycles raise board accuracy, but the refinement gap stays about the same.

---

### Relation to ARC Prize Analysis
These results are consistent with the study by the ARC Prize team ([blog](https://arcprize.org/blog/hrm-analysis), [slides](https://docs.google.com/presentation/d/12IAuVKZXvbW6uCwzDhzN1PBh4fdjypjqyucYzoKJKMg/edit?slide=id.g32b23b2ea24_0_13#slide=id.g32b23b2ea24_0_13)). Their analysis on ARC tasks also found that outer-loop refinement drives performance, while the H/L split is not decisive. The pathfinding experiments here provide an additional supporting data point.

### Iterative Refinement in Action
When trained with more segments, the model reaches higher accuracy and better refines its predictions when given extra inference steps.

The refinement process is visible in how solutions emerge: early steps make broad strokes, while later steps progressively add smaller corrections until the full path is resolved.

| Train Segments = 2 | Train Segments = 4 |
|--------------------|--------------------|
| <img width="1000" height="600" alt="log_HL_hl22_t2_i2" src="https://github.com/user-attachments/assets/32594c32-d601-497d-b61a-b3a09b820436" /> | <img width="1000" height="600" alt="log_HL_hl22_t4_i2" src="https://github.com/user-attachments/assets/c81611a8-8c0f-4134-b502-ce51898cc245" /> |

<img width="400" src="https://github.com/user-attachments/assets/53eee809-6c21-4179-8ab3-9daf4ab74e62" /> <img width="400" src="https://github.com/user-attachments/assets/c4dec441-f1bc-4180-814c-93e1113b367c" />
